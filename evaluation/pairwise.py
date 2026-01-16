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

        # Randomize assignment to A/B for double-blind
        a_is_ft = random.choice([True, False])
        if a_is_ft:
            A_text = q_ft
            B_text = q_gem
        else:
            A_text = q_gem
            B_text = q_ft

        # Gemini judge (sees A/B without source labels)
        gemini_judge_prompt = (
            "下面有兩個題目，請比較並選出較好的題目（以清晰度、正確性、教學價值為準）。請直接輸出 'Winner: A' 或 'Winner: B' 並在上一行提供一句簡短理由。\n\n"
            "A:\n"
            f"{A_text}\n\n"
            "B:\n"
            f"{B_text}\n\n"
        )
        gemini_judge_text, _, _ = gen_from_gemini(gemini_judge_prompt)
        winner_gemini = parse_winner_from_text(gemini_judge_text)

        # Finetuned model judge (also sees A/B without source labels)
        judge_prompt_ft = (
            "下面有兩個題目，請比較並選出較好的題目（以清晰度、正確性、教學價值為準）。請直接輸出 'Winner: A' 或 'Winner: B' 並在上一行提供一句簡短理由。\n\n"
            "A:\n"
            f"{A_text}\n\n"
            "B:\n"
            f"{B_text}\n\n"
        )
        ft_judge_text = gen_from_finetuned(gen, judge_prompt_ft)
        winner_ft = parse_winner_from_text(ft_judge_text)

        # Map judges' choices back to model scores for this round
        round_score_ft = 0
        round_score_gem = 0

        # Gemini judge vote
        if winner_gemini == "A":
            if a_is_ft:
                round_score_ft += 1
            else:
                round_score_gem += 1
        elif winner_gemini == "B":
            if a_is_ft:
                round_score_gem += 1
            else:
                round_score_ft += 1

        # Finetuned judge vote
        if winner_ft == "A":
            if a_is_ft:
                round_score_ft += 1
            else:
                round_score_gem += 1
        elif winner_ft == "B":
            if a_is_ft:
                round_score_gem += 1
            else:
                round_score_ft += 1

        score_ft += round_score_ft
        score_gemini += round_score_gem

        rows.append({
            "round": i + 1,
            "topic": topic,
            "A_is_finetuned": a_is_ft,
            "question_A": A_text,
            "question_B": B_text,
            "gemini_judge": gemini_judge_text.replace("\n", "\\n")[:10000],
            "gemini_choice": winner_gemini,
            "finetuned_judge": ft_judge_text.replace("\n", "\\n")[:10000],
            "finetuned_choice": winner_ft,
            "score_A": (1 if winner_gemini == "A" else 0) + (1 if winner_ft == "A" else 0),
            "score_B": (1 if winner_gemini == "B" else 0) + (1 if winner_ft == "B" else 0),
            "score_finetuned": round_score_ft,
            "score_gemini": round_score_gem,
        })

        print(f"[Pairwise] round {i+1}: A_is_ft={a_is_ft}, ft_round={round_score_ft}, gem_round={round_score_gem}")
        time.sleep(1)

    totals = {"score_finetuned": score_ft, "score_gemini": score_gemini}
    save_rounds_csv(args.output_csv, rows, totals)
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
