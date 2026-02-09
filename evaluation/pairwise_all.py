import argparse
import os
import random
import csv
import itertools
import time
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    parse_winner_from_text,
    cleanup_gpu
)
from GeminiAgent.agent.generator import PROMPT_TEMPLATES, random_topic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories")
    parser.add_argument("--repeats", type=int, default=10, help="Number of topics/questions to generate")
    parser.add_argument("--output_csv", type=str, default="evaluation/pairwise_all_results.csv")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model_paths = [p.rstrip("/\\") for p in args.model_dirs]
    model_names = [os.path.basename(p) for p in model_paths]
    
    # Initialize Scoreboard
    # wins: number of pairwise wins
    # matches: number of pairwise matches played
    scoreboard = {name: {"wins": 0, "ties": 0, "matches": 0} for name in model_names}

    # --- Step 1: Prepare Topics ---
    # We use a fixed set of topics for all models to ensure fair comparison
    tasks = []
    print(f"--- Preparing {args.repeats} Topics ---")
    for i in range(args.repeats):
        topic = random_topic()
        template = random.choice(PROMPT_TEMPLATES)
        # We store the prompt and topic
        tasks.append({
            "id": i+1,
            "topic": topic,
            "prompt": template.format(topic=topic)
        })

    # Store responses: responses[model_name][task_id] = "response text"
    responses = {name: {} for name in model_names}

    # --- Step 2: Generation (Serial Loading to save VRAM) ---
    print("\n=== Phase 1: Generation ===")
    for path, name in zip(model_paths, model_names):
        print(f"\n>>> Loading Model: {name}...")
        try:
            # Check if it is a base model (no adapter config) or PEFT
            # load_finetuned_model handles basic loading, but we need to be careful about VRAM
            # Force 4bit to ensure it fits
            model, tokenizer, gen = load_finetuned_model(path, load_in_4bit=True)
            
            print(f"    Generating responses for {len(tasks)} tasks...")
            for task in tasks:
                prompt = task["prompt"]
                resp = gen_from_finetuned(gen, prompt, temperature=args.temperature, top_p=args.top_p)
                
                if not resp or len(resp.strip()) == 0:
                    resp = "[EMPTY_RESPONSE]"
                
                responses[name][task["id"]] = resp
                # print(f"    Task {task['id']} done.")
            
            print(f"    Model {name} finished.")
            
            # Clean up
            del gen, model, tokenizer
            cleanup_gpu()
            
        except Exception as e:
            print(f"!!! Error loading/running {name}: {e}")
            import traceback
            traceback.print_exc()
            # Fill with errors to avoid crashing later
            for task in tasks:
                if task["id"] not in responses[name]:
                    responses[name][task["id"]] = "[ERROR_GENERATING]"

    # --- Step 3: Pairwise Judging (Round Robin) ---
    print("\n=== Phase 2: Pairwise Judging ===")
    
    # Generate all unique pairs of models
    pairs = list(itertools.combinations(model_names, 2))
    print(f"Total Models: {len(model_names)}")
    print(f"Total Pairs per Topic: {len(pairs)}")
    print(f"Total Judgements: {len(pairs) * len(tasks)}")

    # Prepare CSV Output
    fieldnames = ["task_id", "topic", "model_A", "model_B", "winner", "reason"]
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            t_id = task["id"]
            topic = task["topic"]
            prompt = task["prompt"]
            print(f"\n--- Judging Task {t_id}/{len(tasks)}: {topic} ---")

            for m1, m2 in pairs:
                resp1 = responses[m1][t_id]
                resp2 = responses[m2][t_id]

                # Blind Shuffle
                # We randomly assign m1 to "A" or "B" in the prompt
                is_swapped = random.choice([True, False])
                
                if is_swapped:
                    model_A, model_B = m2, m1
                    text_A, text_B = resp2, resp1
                else:
                    model_A, model_B = m1, m2
                    text_A, text_B = resp1, resp2

                # Judge Prompt (Gemini)
                judge_prompt = (
                    f"You are an expert judge evaluating AI responses on Computer Science topics.\n"
                    f"Topic: {topic}\n"
                    f"Question/Prompt: {prompt}\n\n"
                    f"=== Response A ===\n{text_A}\n\n"
                    f"=== Response B ===\n{text_B}\n\n"
                    f"Instructions:\n"
                    f"1. Compare Response A and Response B based on correctness, depth, and clarity.\n"
                    f"2. Ignore simple formatting differences unless they affect readability.\n"
                    f"3. Select the best response.\n"
                    f"4. First, provide a short explanation (1-2 sentences).\n"
                    f"5. Then, on a new line, output strictly: 'Winner: A', 'Winner: B', or 'Winner: Tie'.\n"
                )

                try:
                    # Judge
                    judge_resp, _, _ = gen_from_gemini(judge_prompt)
                    
                    # Parse Winner
                    # We look for "Winner: A" or "Winner: B"
                    winner_tag = "Tie"
                    if "Winner: A" in judge_resp:
                        winner_tag = "A"
                    elif "Winner: B" in judge_resp:
                        winner_tag = "B"
                    elif "Winner: Tie" in judge_resp:
                        winner_tag = "Tie"
                    
                    # Map back to real model names
                    real_winner = "Tie"
                    if winner_tag == "A":
                        real_winner = model_A
                    elif winner_tag == "B":
                        real_winner = model_B
                    
                    # Update Scoreboard
                    scoreboard[m1]["matches"] += 1
                    scoreboard[m2]["matches"] += 1
                    
                    if real_winner == m1:
                        scoreboard[m1]["wins"] += 1
                    elif real_winner == m2:
                        scoreboard[m2]["wins"] += 1
                    else:
                        scoreboard[m1]["ties"] += 1
                        scoreboard[m2]["ties"] += 1

                    # Log to CSV
                    writer.writerow({
                        "task_id": t_id,
                        "topic": topic,
                        "model_A": model_A,
                        "model_B": model_B,
                        "winner": real_winner, # Logs the name of the winner, or 'Tie'
                        "reason": judge_resp.replace("\n", " ")[:200] + "..." # Truncate log
                    })
                    csvfile.flush()

                    print(f"   {m1} vs {m2} -> Winner: {real_winner}")
                    # Sleep slightly to avoid API flooding if needed
                    time.sleep(1) 

                except Exception as e:
                    print(f"Error judging {m1} vs {m2}: {e}")

    # --- Step 4: Final Report ---
    print("\n" + "="*40)
    print("FINAL PAIRWISE TOURNAMENT RESULTS")
    print("="*40)
    print(f"{'Model Name':<30} | {'Wins':<5} | {'Ties':<5} | {'Matches':<8} | {'Win Rate':<8}")
    print("-" * 70)
    
    # Sort by Wins desc
    sorted_models = sorted(scoreboard.items(), key=lambda x: x[1]['wins'], reverse=True)
    
    for name, stats in sorted_models:
        wins = stats['wins']
        ties = stats['ties']
        matches = stats['matches']
        win_rate = (wins / matches * 100) if matches > 0 else 0
        print(f"{name:<30} | {wins:<5} | {ties:<5} | {matches:<8} | {win_rate:.1f}%")
    
    # Highlight Champion
    print("="*40)
    champion = sorted_models[0][0]
    print(f"🏆 CHAMPION: {champion}")
    print("="*40)

if __name__ == "__main__":
    main()
