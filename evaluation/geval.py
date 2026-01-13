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

    for i in range(args.repeats):
        topic = random_topic()
        # instruct both models to produce one MCQ question about the topic
        prompt = f"請用繁體中文根據主題「{topic}」出一道單選題（四選一），包含題目、選項(A/B/C/D)、答案以及簡短解析。"

        # fine-tuned model generation
        q_ft = gen_from_finetuned(gen, prompt)

        # gemini generation via existing generator (we call gen_from_gemini with similar prompt)
        q_gem, _, _ = gen_from_gemini(prompt)

        # Now ask Gemini to score both with CoT: request reasoning then integer 1-10
        judge_prompt = (
            "你是評分員。請以 Chain-of-Thought (先說明理由) 的方式，分別對下面兩份試題從 1 到 10 (整數) 評分，10 為最佳。\n\n"
            f"Topic: {topic}\n\n"
            "A: (fine-tuned model)\n"
            f"{q_ft}\n\n"
            "B: (Gemini)\n"
            f"{q_gem}\n\n"
            "請先給出每題的詳細評分理由，再在各自理由最後一行回報 'Score: X'（X 為 1-10 的整數）。只要數字即可作為分數行的結尾。"
        )

        judge_text, _, _ = gen_from_gemini(judge_prompt)

        # attempt to extract two scores (first for A then for B)
        # simple approach: split by lines and extract first two integers found sequentially
        scores = []
        for part in judge_text.splitlines():
            s = parse_score_from_text(part)
            if s != -1:
                scores.append(s)
            if len(scores) >= 2:
                break

        score_a = scores[0] if len(scores) >= 1 else -1
        score_b = scores[1] if len(scores) >= 2 else -1

        if score_a > 0:
            total_ft += score_a
        if score_b > 0:
            total_gemini += score_b

        rows.append({
            "round": i + 1,
            "topic": topic,
            "question_finetuned": q_ft,
            "question_gemini": q_gem,
            "score_finetuned": score_a,
            "score_gemini": score_b,
            "judge_raw": judge_text.replace("\n", "\\n")[:10000],
        })

        print(f"[G-Eval] round {i+1}: ft={score_a}, gemini={score_b}")
        time.sleep(1)

    totals = {"total_finetuned": total_ft, "total_gemini": total_gemini}
    save_rounds_csv(args.output_csv, rows, totals)
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
