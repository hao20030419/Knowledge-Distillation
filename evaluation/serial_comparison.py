import argparse
import os
import json
import time
import random
import csv
import torch
import gc
import re
# 假設這些工具函數都放在 evaluation/utils.py 或你指定的路徑
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    extract_multi_scores,  # 建議使用我剛才提供的新版解析函式
    cleanup_gpu
)
from GeminiAgent.agent.generator import PROMPT_TEMPLATES, random_topic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True, help="List of model directories")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/results.csv")
    parser.add_argument("--responses_dir", type=str, default="evaluation/responses")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.responses_dir, exist_ok=True)
    
    model_paths = [p.rstrip("/\\") for p in args.model_dirs]
    model_names = [os.path.basename(p) for p in model_paths]

    # --- Step 1: Scenario Preparation (固定每一輪的主題) ---
    scenarios = []
    for i in range(args.repeats):
        topic = random_topic()
        template = random.choice(PROMPT_TEMPLATES)
        scenarios.append({
            "round": i + 1,
            "topic": topic,
            "full_prompt": template.format(topic=topic)
        })

    # --- Step 2: Generation (模型生成階段) ---
    for path, name in zip(model_paths, model_names):
        # 檢查是否已經生成過，若有則跳過 (選做)
        print(f"\n>>> Loading Model: {name}")
        try:
            model, tokenizer, gen = load_finetuned_model(path, load_in_4bit=True)
            for sc in scenarios:
                # 確保生成時使用的是該輪特定的 prompt
                resp = gen_from_finetuned(gen, sc["full_prompt"], temperature=args.temperature, top_p=args.top_p)
                
                fname = f"round{sc['round']}_model_{name}.txt"
                with open(os.path.join(args.responses_dir, fname), "w", encoding="utf-8") as f:
                    f.write(resp)
            
            # 釋放顯存
            del gen, model, tokenizer
            cleanup_gpu()
        except Exception as e:
            print(f"Error on model {name}: {e}")

    # --- Step 3: Judging (評審階段) ---
    fieldnames = ["round", "topic", "prompt"] + [f"score_{n}" for n in model_names] + ["judge_raw_response"]
    totals = {n: 0 for n in model_names}

    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sc in scenarios:
            # 打亂順序以避免位置偏差 (Position Bias)
            shuffled_indices = list(range(len(model_names)))
            random.shuffle(shuffled_indices)
            
            judge_content = ""
            idx_to_name = {} 
            
            for i, s_idx in enumerate(shuffled_indices):
                m_name = model_names[s_idx]
                m_label = i + 1
                idx_to_name[m_label] = m_name
                
                fpath = os.path.join(args.responses_dir, f"round{sc['round']}_model_{m_name}.txt")
                if os.path.exists(fpath):
                    with open(fpath, "r", encoding="utf-8") as rf:
                        content = rf.read()
                else:
                    content = "No Output"
                
                judge_content += f"=== Model {m_label} ===\n{content}\n\n"

            # 【關鍵修正】: 使用 sc['topic'] 而非全局變數 topic
            judge_prompt = (
                f"你是一位嚴苛的學術評審。請以 Chain-of-Thought (先說明理由) 的方式，針對主題「{sc['topic']}」對這 {len(model_paths)} 個模型的試題進行「強制排名評分」。\n\n"
                f"【評分軍規】\n"
                f"1. 僅看題幹：無視解析、答案、排版、贅語或幻覺。\n"
                f"2. 強制分差：第一名(最佳)必給10分，最後一名必給1分。嚴禁給予相近或重複分數，必須拉大分差。\n"
                f"3. 核心指標：正確性(邏輯無誤) + 深度(高階應用)。\n\n"
                f"【待評內容】\n{judge_content}"
                f"請先說明理由 (CoT)，最後務必以「Final Ratings:」標籤開頭並按下列格式結尾：\n"
                + "\n".join([f"Model {i+1} Score: X" for i in range(len(model_paths))])
            )

            print(f"--- Round {sc['round']} Judging Topic: {sc['topic']} ---")
            raw_judge_resp, _, _ = gen_from_gemini(judge_prompt)
            
            # 使用更魯棒的解析方式
            extracted_scores = extract_multi_scores(raw_judge_resp, len(model_paths))
            
            row = {
                "round": sc["round"], 
                "topic": sc["topic"], 
                "prompt": sc["full_prompt"], 
                "judge_raw_response": raw_judge_resp
            }
            
            # 映射分數回 CSV
            for m_label, m_real_name in idx_to_name.items():
                s = extracted_scores.get(m_label, -1)
                row[f"score_{m_real_name}"] = s
                if s > 0: totals[m_real_name] += s

            writer.writerow(row)
            f.flush()
            print(f"Parsed Scores: {extracted_scores}")

    print("\n" + "="*30)
    print("Evaluation Complete.")
    print("Final Totals:", totals)
    print("="*30)

if __name__ == "__main__":
    main()