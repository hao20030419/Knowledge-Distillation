import argparse
import time
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
    parser.add_argument("--model_dir", type=str, required=True, help="path to fine-tuned model dir")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/pairwise_results.csv")
    args = parser.parse_args()

    model, tokenizer, gen = load_finetuned_model(args.model_dir)

    rows = []
    score_ft = 0
    score_gemini = 0

    for i in range(args.repeats):
        topic = random_topic()
        prompt = f"請用繁體中文根據主題「{topic}」出一道單選題（四選一），包含題目、選項(A/B/C/D)、答案以及簡短解析。"

        q_ft = gen_from_finetuned(gen, prompt)
        q_gem, _, _ = gen_from_gemini(prompt)

        # Now ask both models to choose which question is better (A = finetuned, B = gemini)
        # First: Gemini judge
        gemini_judge_prompt = (
            f"下面有兩個題目，請比較並選出較好的題目（以清晰度、正確性、教學價值為準）。請直接輸出 'Winner: A' 或 'Winner: B' 並在上一行提供一句簡短理由。\n\n"
            "A:\n"
            f"{q_ft}\n\n"
            "B:\n"
            f"{q_gem}\n\n"
        )
        gemini_judge_text, _, _ = gen_from_gemini(gemini_judge_prompt)
        winner_gemini = parse_winner_from_text(gemini_judge_text)

        # Second: finetuned model judge (we prompt it similarly)
        judge_prompt_ft = (
            f"下面有兩個題目，請比較並選出較好的題目（以清晰度、正確性、教學價值為準）。請直接輸出 'Winner: A' 或 'Winner: B' 並在上一行提供一句簡短理由。\n\n"
            "A:\n"
            f"{q_ft}\n\n"
            "B:\n"
            f"{q_gem}\n\n"
        )
        ft_judge_text = gen_from_finetuned(gen, judge_prompt_ft)
        winner_ft = parse_winner_from_text(ft_judge_text)

        # tally: each judge gives 1 point to winner
        if winner_gemini == "A":
            score_ft += 1
        elif winner_gemini == "B":
            score_gemini += 1

        if winner_ft == "A":
            score_ft += 1
        elif winner_ft == "B":
            score_gemini += 1

        rows.append({
            "round": i + 1,
            "topic": topic,
            "question_finetuned": q_ft,
            "question_gemini": q_gem,
            "gemini_judge": gemini_judge_text.replace("\n", "\\n")[:10000],
            "gemini_choice": winner_gemini,
            "finetuned_judge": ft_judge_text.replace("\n", "\\n")[:10000],
            "finetuned_choice": winner_ft,
        })

        print(f"[Pairwise] round {i+1}: gemini->{winner_gemini}, ft->{winner_ft}")
        time.sleep(1)

    totals = {"score_finetuned": score_ft, "score_gemini": score_gemini}
    save_rounds_csv(args.output_csv, rows, totals)
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
