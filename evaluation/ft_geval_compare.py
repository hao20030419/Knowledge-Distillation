import argparse
import time
import random
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
    parser.add_argument("--before_dir", type=str, required=True, help="path to model before fine-tune")
    parser.add_argument("--after_dir", type=str, required=True, help="path to model after fine-tune")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/ft_geval_results.csv")
    args = parser.parse_args()

    _, _, gen_before = load_finetuned_model(args.before_dir)
    _, _, gen_after = load_finetuned_model(args.after_dir)

    rows = []
    total_before = 0
    total_after = 0

    for i in range(args.repeats):
        topic = random_topic()
        prompt = f"請用繁體中文根據主題「{topic}」出一道單選題（四選一），包含題目、選項(A/B/C/D)、答案以及簡短解析。"

        q_before = gen_from_finetuned(gen_before, prompt)
        q_after = gen_from_finetuned(gen_after, prompt)

        # double-blind randomization: A may be 'after' or 'before'
        a_is_after = random.choice([True, False])
        if a_is_after:
            A_text = q_after
            B_text = q_before
        else:
            A_text = q_before
            B_text = q_after

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

        # map back
        if a_is_after:
            score_after = score_a
            score_before = score_b
        else:
            score_before = score_a
            score_after = score_b

        if score_before > 0:
            total_before += score_before
        if score_after > 0:
            total_after += score_after

        rows.append({
            "round": i + 1,
            "topic": topic,
            "A_is_after": a_is_after,
            "question_A": A_text,
            "question_B": B_text,
            "score_A": score_a,
            "score_B": score_b,
            "score_before": score_before,
            "score_after": score_after,
            "judge_raw": judge_text.replace("\n", "\\n")[:10000],
        })

        print(f"[FT G-Eval] round {i+1}: A_is_after={a_is_after}, before={score_before}, after={score_after}")
        time.sleep(1)

    totals = {"total_before": total_before, "total_after": total_after}
    save_rounds_csv(args.output_csv, rows, totals)
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
