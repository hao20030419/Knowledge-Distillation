import argparse
import time
import re
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
)
from GeminiAgent.agent.generator import random_topic
from GPTagent.agent.generator import gpt5_client

def gen_from_gpt(prompt):
    """
    使用最新的 GPT-5.2 進行生成。
    """
    try:
        # 建議確保 gpt5_client 是最新的 OpenAI() 實例
        response = gpt5_client.chat.completions.create(
            model="gpt-5.2",  # 更新為最新的穩定版
            messages=[
                {"role": "developer", "content": "你是一個專業的學術評分員。"}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, # 評分任務建議溫度調低一點，增加穩定性
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT-5: {e}")
        return ""

def parse_score_from_text(text: str) -> int:
    """從 'Score: X' 或類似字樣中解析分數 (1-10)"""
    # 針對 "Score: 8" 或 "Score: 10" 的格式
    match = re.search(r"Score[:\s]*(\d+)", text, re.IGNORECASE)
    if match:
        try:
            val = int(match.group(1))
            if 1 <= val <= 10:
                return val
        except ValueError:
            pass
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="path to fine-tuned model dir")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/geval_results.csv")
    parser.add_argument("--temperature", type=float, default=0.7, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="generation top_p")
    args = parser.parse_args()

    model, tokenizer, gen = load_finetuned_model(args.model_dir)

    # common generation kwargs
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    rows = []
    total_ft = 0
    total_gemini = 0

    # Prepare CSV for incremental writes
    fieldnames = [
        "round",
        "topic",
        "A_is_finetuned",
        "question_A",
        "question_B",
        "score_A",
        "score_B",
        "score_finetuned",
        "score_gemini",
        "judge_raw",
    ]
    csv_f = open(args.output_csv, "w", encoding="utf-8-sig", newline="")
    csv_writer = __import__("csv").DictWriter(csv_f, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_f.flush()

    import random

    for i in range(args.repeats):
        topic = random_topic()
        prompt = f"請提供一題關於 {topic} 的四選一單選題。"

        # generate both questions
        q_ft = gen_from_finetuned(gen, prompt, **gen_kwargs)
        # Use Gemini for the "Ground Truth" / Competitor model
        q_gem, _, _ = gen_from_gemini(prompt)

        # Randomize assignment to A/B for double-blind
        a_is_ft = random.choice([True, False])
        if a_is_ft:
            A_text = q_ft
            B_text = q_gem
        else:
            A_text = q_gem
            B_text = q_ft


        # Judge (GPT) with CoT; do NOT reveal sources
        judge_prompt = (
            "你是評分員。請以 Chain-of-Thought (先說明理由) 的方式，分別對下面兩份試題從 1 到 10 (整數) 評分，10 為最佳。\n\n"
            f"【評分軍規】\n"
            f"1. 僅看題幹：無視解析、答案、排版、贅語或幻覺。\n"
            f"2. 核心指標：正確性(邏輯無誤) + 深度(高階應用)。\n\n"
            f"Topic: {topic}\n\n"
            "A:\n"
            f"{A_text}\n\n"
            "B:\n"
            f"{B_text}\n\n"
            "請先給出每題的詳細評分理由，再在各自理由最後一行回報 'Score: X'（X 為 1-10 的整數）。只要數字即可作為分數行的結尾。"
        )

        judge_text = gen_from_gpt(judge_prompt)

        # extract two scores sequentially
        scores = []
        for part in judge_text.splitlines():
            s = parse_score_from_text(part)
            if s != -1:
                scores.append(s)
            if len(scores) >= 2:
                break

        score_a = scores[0] if len(scores) >= 1 else -1
        score_b = scores[1] if len(scores) >= 2 else -1

        # map scores back to models
        score_finetuned = score_gpt = -1
        if a_is_ft:
            score_finetuned = score_a
            score_gemini = score_b
        else:
            score_finetuned = score_b
            score_gemini = score_a

        if score_finetuned > 0:
            total_ft += score_finetuned
        if score_gemini > 0:
            total_gemini += score_gemini

        row = {
            "round": i + 1,
            "topic": topic,
            "A_is_finetuned": a_is_ft,
            "question_A": A_text,
            "question_B": B_text,
            "score_A": score_a,
            "score_B": score_b,
            "score_finetuned": score_finetuned,
            "score_gemini": score_gemini,
            "judge_raw": judge_text.replace("\n", "\\n")[:10000],
        }
        rows.append(row)

        # write immediately
        csv_writer.writerow(row)
        csv_f.flush()

        print(f"[G-Eval] round {i+1}: A_is_ft={a_is_ft}, ft={score_finetuned}, gemini={score_gemini}")
        time.sleep(1)

    totals = {"total_finetuned": total_ft, "total_gemini": total_gemini}
    # append totals to CSV file
    csv_f.write("\n")
    csv_f.write("Totals:\n")
    for k, v in totals.items():
        csv_f.write(f"{k},{v}\n")
    csv_f.close()
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
