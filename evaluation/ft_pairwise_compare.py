import argparse
import time
import random
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    parse_winner_from_text,
    random_topic,
    save_rounds_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_dir", type=str, required=True, help="path to model before fine-tune")
    parser.add_argument("--after_dir", type=str, required=True, help="path to model after fine-tune")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/ft_pairwise_results.csv")
    parser.add_argument("--temperature", type=float, default=0.7, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="generation top_p")
    args = parser.parse_args()

    _, _, gen_before = load_finetuned_model(args.before_dir)
    _, _, gen_after = load_finetuned_model(args.after_dir)

    # common generation kwargs
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    rows = []
    score_before = 0
    score_after = 0

    # Prepare CSV for incremental writes
    fieldnames = [
        "round",
        "topic",
        "A_is_after",
        "question_A",
        "question_B",
        "gemini_judge",
        "gemini_choice",
        "score_before",
        "score_after",
    ]
    csv_f = open(args.output_csv, "w", encoding="utf-8-sig", newline="")
    csv_writer = __import__("csv").DictWriter(csv_f, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_f.flush()

    for i in range(args.repeats):
        topic = random_topic()
        prompt = f"請提供一題關於 {topic} 的四選一單選題。"

        q_before = gen_from_finetuned(gen_before, prompt, **gen_kwargs)
        q_after = gen_from_finetuned(gen_after, prompt, **gen_kwargs)

        # double-blind randomization
        a_is_after = random.choice([True, False])
        if a_is_after:
            A_text = q_after
            B_text = q_before
        else:
            A_text = q_before
            B_text = q_after

        gemini_judge_prompt = (
            "下面有兩個題目，請比較並選出較好的題目（以清晰度、正確性、教學價值為準）。請直接輸出 'Winner: A' 或 'Winner: B' 並在上一行提供一句簡短理由。\n\n"
            "A:\n"
            f"{A_text}\n\n"
            "B:\n"
            f"{B_text}\n\n"
        )

        gemini_judge_text, _, _ = gen_from_gemini(gemini_judge_prompt)
        winner_gemini = parse_winner_from_text(gemini_judge_text)

        # Map Gemini choice back to before/after scores
        round_before = 0
        round_after = 0
        if winner_gemini == "A":
            if a_is_after:
                round_after += 1
            else:
                round_before += 1
        elif winner_gemini == "B":
            if a_is_after:
                round_before += 1
            else:
                round_after += 1

        score_before += round_before
        score_after += round_after

        row = {
            "round": i + 1,
            "topic": topic,
            "A_is_after": a_is_after,
            "question_A": A_text,
            "question_B": B_text,
            "gemini_judge": gemini_judge_text.replace("\n", "\\n")[:10000],
            "gemini_choice": winner_gemini,
            "score_before": round_before,
            "score_after": round_after,
        }
        rows.append(row)

        # write immediately
        csv_writer.writerow(row)
        csv_f.flush()

        print(f"[FT Pairwise] round {i+1}: A_is_after={a_is_after}, before_round={round_before}, after_round={round_after}")
        time.sleep(1)

    totals = {"score_before": score_before, "score_after": score_after}
    csv_f.write("\n")
    csv_f.write("Totals:\n")
    for k, v in totals.items():
        csv_f.write(f"{k},{v}\n")
    csv_f.close()
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
