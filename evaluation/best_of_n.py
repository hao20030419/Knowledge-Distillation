import argparse
import os
import json
import random
import csv
import time
import re
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    cleanup_gpu
)
from GeminiAgent.agent.generator import random_topic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories")
    parser.add_argument("--repeats", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--output_csv", type=str, default="evaluation/best_of_n_results.csv")
    parser.add_argument("--prompts_file", type=str, default="GeminiAgent/agent/prompts.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model_paths = [p.rstrip("/\\") for p in args.model_dirs]
    model_names = [os.path.basename(p) for p in model_paths]
    
    print(f"Loading prompts from {args.prompts_file}...")
    try:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompt_templates = json.load(f)
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        return

    # Initialize Scoreboard
    scoreboard = {name: 0 for name in model_names}

    # --- Step 1: Prepare Topics & Prompts ---
    tasks = []
    print(f"--- Preparing {args.repeats} Tasks ---")
    for i in range(args.repeats):
        topic = random_topic()
        template = random.choice(prompt_templates)
        full_prompt = template.format(topic=topic)
        tasks.append({
            "id": i+1,
            "topic": topic,
            "prompt": full_prompt
        })

    # Store responses: responses[task_id][model_name] = text
    responses_storage = {t["id"]: {} for t in tasks}

    # --- Step 2: Generation (Serial Loading) ---
    print("\n=== Phase 1: Generation ===")
    for path, name in zip(model_paths, model_names):
        print(f"\n>>> Loading Model: {name}...")
        try:
            model, tokenizer, gen = load_finetuned_model(path, load_in_4bit=True)
            
            print(f"    Generating responses for {len(tasks)} tasks...")
            for task in tasks:
                prompt = task["prompt"]
                resp = gen_from_finetuned(gen, prompt, temperature=args.temperature, top_p=args.top_p)
                
                if not resp or len(resp.strip()) == 0:
                    resp = "[EMPTY_RESPONSE]"
                
                responses_storage[task["id"]][name] = resp
            
            print(f"    Model {name} finished.")
            
            del gen, model, tokenizer
            cleanup_gpu()
            
        except Exception as e:
            print(f"!!! Error loading/running {name}: {e}")
            import traceback
            traceback.print_exc()
            for task in tasks:
                responses_storage[task["id"]][name] = "[ERROR]"

    # --- Step 3: Judging (Best of N) ---
    print("\n=== Phase 2: Best-of-N Judging ===")
    
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Updated CSV fields
    fieldnames = ["task_id", "topic", "prompt", "winner_model", "judge_reason"] + [f"resp_{name}" for name in model_names]

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            t_id = task["id"]
            topic = task["topic"]
            prompt = task["prompt"]
            
            print(f"\n--- Judging Task {t_id}/{len(tasks)}: {topic} ---")
            
            # Identify models participating in this round
            models_in_round = model_names.copy()
            
            # Shuffle models to create Blind Test
            random.shuffle(models_in_round)
            
            # Construct Judge Prompt
            judge_content = ""
            # Mapping from 'Model X' label to real model name
            label_map = {} 
            
            for idx, m_name in enumerate(models_in_round):
                label = f"Model {idx+1}"
                label_map[label] = m_name
                resp_text = responses_storage[t_id][m_name]
                judge_content += f"=== {label} ===\n{resp_text}\n\n"

            gemini_prompt = (
                f"你是一位資訊工程(CS)領域的專家評審。\n"
                f"評測主題: {topic}\n"
                f"問題提示: {prompt}\n\n"
                f"以下是 {len(models_in_round)} 個不同 AI 模型的回答。 "
                f"請確認它們是否遵循了指令（通常要求生成一題四選一的單選題）。\n"
                f"請根據 正確性（CS知識是否精確？）、清晰度（問題/回答的表達是否流暢？）以及 整體品質 來比較這些回答。\n\n"
                f"{judge_content}"
                f"評分指令：\n"
                f"1. 請簡短說明你的比較理由及分析。\n"
                f"2. 選出表現 唯一最好 的模型。\n"
                f"3. 請務必在最後一行嚴格輸出：'Winner: Model X'（其中 X 為該模型的編號）。\n"
            )
            
            try:
                judge_resp, _, _ = gen_from_gemini(gemini_prompt)
                
                # Parse Winner
                # Regex for "Winner: Model <digits>"
                match = re.search(r"Winner:\s*(Model\s*\d+)", judge_resp, re.IGNORECASE)
                
                if match:
                    winner_label = match.group(1) # e.g. "Model 3" or "model 3"
                    # Normalize label format to match keys in label_map (Model X)
                    # We assume regex captured "Model <digits>"
                    # Let's clean it up to be sure
                    parts = winner_label.split()
                    clean_label = f"Model {parts[-1]}" # "Model", "3"
                    
                    real_winner = label_map.get(clean_label, "Unknown")
                    if real_winner != "Unknown":
                        scoreboard[real_winner] += 1
                else:
                    real_winner = "Parse Error"
                
                print(f"   Winner: {real_winner} (Label: {match.group(1) if match else 'None'})")

                # Build row
                row = {
                    "task_id": t_id,
                    "topic": topic,
                    "prompt": prompt,
                    "winner_model": real_winner,
                    "judge_reason": judge_resp
                }
                # Fill model responses
                for m_name in model_names:
                    row[f"resp_{m_name}"] = responses_storage[t_id][m_name]
                
                writer.writerow(row)
                csvfile.flush()
                time.sleep(1)

            except Exception as e:
                print(f"Error judging task {t_id}: {e}")

    # --- Step 4: Final Report ---
    print("\n" + "="*40)
    print("BEST-OF-N TOURNAMENT RESULTS")
    print("="*40)
    print(f"{'Model Name':<30} | {'Wins':<5} | {'Win Rate':<8}")
    print("-" * 60)
    
    sorted_models = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)
    total_tasks = len(tasks)
    
    for name, wins in sorted_models:
        rate = (wins / total_tasks * 100) if total_tasks > 0 else 0
        print(f"{name:<30} | {wins:<5} | {rate:.1f}%")
    
    print("="*40)
    print(f"🏆 CHAMPION: {sorted_models[0][0]}")
    print("="*40)

if __name__ == "__main__":
    main()
