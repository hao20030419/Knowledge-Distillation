import argparse
import time
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    parse_score_from_text,
    random_topic,
    save_rounds_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="path to fine-tuned model dir")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/geval_results.csv")
    args = parser.parse_args()

    model, tokenizer, gen = load_finetuned_model(args.model_dir)

    rows = []
    total_ft = 0
    total_gemini = 0

    import random

    for i in range(args.repeats):
        topic = random_topic()
        prompt = f"請用繁體中文根據主題「{topic}」出一道單選題（四選一），包含題目、選項(A/B/C/D)、答案以及簡短解析。"

        # generate both questions
        q_ft = gen_from_finetuned(gen, prompt)
        q_gem, _, _ = gen_from_gemini(prompt)

        # Randomize assignment to A/B for double-blind
        a_is_ft = random.choice([True, False])
        if a_is_ft:
            A_text = q_ft
            B_text = q_gem
        else:
            A_text = q_gem
            B_text = q_ft

        # Judge (Gemini) with CoT; do NOT reveal sources
        judge_prompt = (
            "你是評分員。請以 Chain-of-Thought (先說明理由) 的方式，分別對下面兩份試題從 1 到 10 (整數) 評分，10 為最佳。\n\n"
            f"Topic: {topic}\n\n"
            "A:\n"
            f"{A_text}\n\n"
            "B:\n"
            f"{B_text}\n\n"
            "請先給出每題的詳細評分理由，再在各自理由最後一行回報 'Score: X'（X 為 1-10 的整數）。只要數字即可作為分數行的結尾。"
        )

        judge_text, _, _ = gen_from_gemini(judge_prompt)

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
        score_finetuned = score_gemini = -1
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

        rows.append({
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
        })

        print(f"[G-Eval] round {i+1}: A_is_ft={a_is_ft}, ft={score_finetuned}, gemini={score_gemini}")
        time.sleep(1)

    totals = {"total_finetuned": total_ft, "total_gemini": total_gemini}
    save_rounds_csv(args.output_csv, rows, totals)
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
