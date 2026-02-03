import argparse
import os
import json
import time
import random
import csv
import torch
import gc
import re
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    parse_score_from_text,
    random_topic,
)
from GeminiAgent.agent.generator import PROMPT_TEMPLATES

def get_prompts_list():
    """Load prompts from JSON or fallback to default list."""
    # Try to find prompts.json in GeminiAgent/agent/prompts.json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "GeminiAgent", "agent", "prompts.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception:
            pass
    return PROMPT_TEMPLATES

def cleanup_gpu():
    """Force garbage collection and empty CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories to evaluate sequentially")
    parser.add_argument("--repeats", type=int, default=10, help="Number of test rounds")
    parser.add_argument("--output_csv", type=str, default="evaluation/serial_comparison_results.csv")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    prompts_pool = get_prompts_list()
    
    # --- Step 1: Prepare Scenarios (Topic + Prompt) ---
    print(f"=== Step 1: Preparing {args.repeats} test scenarios ===")
    scenarios = []
    for i in range(args.repeats):
        topic = random_topic()
        template = random.choice(prompts_pool)
        prompt_text = template.format(topic=topic)
        scenarios.append({
            "round": i + 1,
            "topic": topic,
            "template": template,
            "prompt": prompt_text,
            "responses": {} # Will store { "model_path": "response_text" }
        })

    # --- Step 2: Serial Generation (Load one model, gen all, unload) ---
    print(f"=== Step 2: Generating responses from {len(args.model_dirs)} models sequentially ===")
    
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    for model_path in args.model_dirs:
        model_name = os.path.basename(model_path.rstrip("/\\"))
        print(f"\n--- Loading Model: {model_name} ({model_path}) ---")
        
        try:
            # Load
            model, tokenizer, gen = load_finetuned_model(
                model_path, 
                load_in_8bit=args.load_in_8bit, 
                load_in_4bit=args.load_in_4bit
            )
            
            # Generate for all scenarios
            print(f"Generating {len(scenarios)} responses...")
            for idx, sc in enumerate(scenarios):
                resp = gen_from_finetuned(gen, sc["prompt"], **generation_kwargs)
                sc["responses"][model_path] = resp
                # Print each model's response to the console for inspection
                try:
                    print(f"[{model_name}] Round {idx+1} response:\n{resp}\n{'-'*80}")
                except Exception:
                    # Fallback if response contains non-printable characters
                    print(f"[{model_name}] Round {idx+1} response: (unable to display raw response)\n{'-'*80}")
                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx + 1}/{len(scenarios)} queries.")

        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
        finally:
            # Unload explicitly
            print(f"Unloading {model_name}...")
            if 'gen' in locals(): del gen
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            cleanup_gpu()
            time.sleep(2) # brief pause to let memory settle

    # --- Step 3: Judge with Gemini ---
    print(f"\n=== Step 3: Judging with Gemini ===")
    
    # Open CSV for writing
    # Construct field names: round, topic, prompt, score_model1, score_model2..., judge_reason
    # We use model directory names as column headers
    model_headers = [os.path.basename(p.rstrip("/\\")) for p in args.model_dirs]
    fieldnames = ["round", "topic", "prompt"] + [f"score_{m}" for m in model_headers] + ["judge_reason", "judge_raw_response"]
    
    dir_name = os.path.dirname(args.output_csv)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        totals = {m: 0 for m in model_headers}

        for sc in scenarios:
            r_idx = sc["round"]
            topic = sc["topic"]
            
            # Construct Judge Prompt
            # We anonymize models as Model 1, Model 2... to keep judging fair (though Gemini naming implies model type anyway)
            judge_prompt = (
                f"你是評分員。請針對主題「{topic}」的試題生成結果，對下列 {len(model_headers)} 個模型的輸出進行評分 (1-10 分，10 為最佳)。\n"
                f"評分標準：題目正確性、選項合理性、解析完整度、是否符合 Prompt 要求。\n\n"
                f"User Prompt: {sc['prompt']}\n\n"
            )

            for idx, model_path in enumerate(args.model_dirs):
                m_label = model_headers[idx]
                resp_text = sc["responses"].get(model_path, "(No Output)")
                judge_prompt += f"=== Model {idx+1} ({m_label}) ===\n{resp_text}\n\n"

            judge_prompt += (
                "請先對每個模型給出簡短評語，最後在結尾列出分數，格式如下：\n"
            )
            for idx, m_label in enumerate(model_headers):
                judge_prompt += f"Model {idx+1} Score: X\n"
            
            # Call Gemini
            judge_response, _, _ = gen_from_gemini(judge_prompt)
            
            # Parse scores
            # Custom parsing to robustly find "Model N Score: X" or similar lines
            row = {
                "round": r_idx,
                "topic": topic,
                "prompt": sc["prompt"],
                "judge_reason": judge_response.replace("\n", "\\n")[:5000],
                "judge_raw_response": judge_response
            }
            
            # Heuristic parsing for multiple scores
            # expecting lines like "Model 1 Score: 8"
            for idx, m_label in enumerate(model_headers):
                score = -1
                # Try specific pattern for this model
                # Pattern: "Model {idx+1} Score: {num}" 
                # or just look for "Score: {num}" in the paragraph mentioned model... strict is better.
                
                # Regex to search for "Model X ... Score: Y"
                # This simple regex assumes the standard requested format.
                pattern = fr"Model {idx+1}.*?Score\s*[:：-]?\s*(\d{{1,2}})"
                match = re.search(pattern, judge_response, re.IGNORECASE | re.DOTALL)
                
                if match:
                    try:
                        score = int(match.group(1))
                    except: pass
                
                # Fallback: if user prompt structure changes or Gemini ignores format
                if score == -1:
                    # Try finding just lines starting with Model X Score
                    lines = judge_response.splitlines()
                    for ln in lines:
                        if f"Model {idx+1}" in ln and "Score" in ln:
                            s = parse_score_from_text(ln)
                            if s != -1:
                                score = s
                                break
                
                row[f"score_{m_label}"] = score
                if score > 0:
                    totals[m_label] += score

            writer.writerow(row)
            f.flush()
            print(f"[Round {r_idx}] Judge finished. Scores: {[row.get(f'score_{m}', -1) for m in model_headers]}")
            time.sleep(1) # rate limit safety

        # Write totals
        f.write("\n")
        f.write("Totals:\n")
        for m in model_headers:
            f.write(f"{m},{totals[m]}\n")
            
    print(f"\nAll Done! Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
