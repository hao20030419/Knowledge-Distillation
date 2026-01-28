import argparse
import time
import random
from evaluation.utils import (
    load_finetuned_model,
    gen_from_finetuned,
    gen_from_gemini,
    gen_from_gpt,
    parse_winner_from_text,
    random_topic,
    save_rounds_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="path to fine-tuned model dir")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/pairwise_results.csv")
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
    score_ft = 0
    score_gemini = 0

    # Prepare CSV for incremental writes
    fieldnames = [
        "round",
        "topic",
        "A_is_finetuned",
        "question_A",
        "question_B",
        "gpt_judge",
        "gpt_choice",
        "score_A",
        "score_B",
        "score_finetuned",
        "score_gemini",
    ]
    csv_f = open(args.output_csv, "w", encoding="utf-8-sig", newline="")
    csv_writer = __import__("csv").DictWriter(csv_f, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_f.flush()

    for i in range(args.repeats):
        topic = random_topic()
        prompt = f"請用繁體中文根據主題「{topic}」出一道單選題（四選一），包含題目、選項(A/B/C/D)、答案以及簡短解析。"

        q_ft = gen_from_finetuned(gen, prompt, **gen_kwargs)
        q_gem, _, _ = gen_from_gemini(prompt)

        # Randomize assignment to A/B for double-blind
        a_is_ft = random.choice([True, False])
        if a_is_ft:
            A_text = q_ft
            B_text = q_gem
        else:
            A_text = q_gem
            B_text = q_ft

        # GPT judge (sees A/B without source labels)
        gemini_judge_prompt = (
            "下面有兩個題目，請比較並選出較好的題目（以清晰度、正確性、教學價值為準）。請直接輸出 'Winner: A' 或 'Winner: B' 並在上一行提供理由。\n\n"
            "A:\n"
            f"{A_text}\n\n"
            "B:\n"
            f"{B_text}\n\n"
        )
        gemini_judge_text = gen_from_gpt(gemini_judge_prompt)
        winner_gemini = parse_winner_from_text(gemini_judge_text)

        # Map GPT judge choice back to model scores for this round
        round_score_ft = 0
        round_score_gem = 0
        if winner_gemini == "A":
            if a_is_ft:
                round_score_ft = 1
            else:
                round_score_gem = 1
        elif winner_gemini == "B":
            if a_is_ft:
                round_score_gem = 1
            else:
                round_score_ft = 1

        score_ft += round_score_ft
        score_gemini += round_score_gem

        row = {
            "round": i + 1,
            "topic": topic,
            "A_is_finetuned": a_is_ft,
            "question_A": A_text,
            "question_B": B_text,
            "gpt_judge": gemini_judge_text.replace("\n", "\\n")[:10000],
            "gpt_choice": winner_gemini,
            "score_finetuned": round_score_ft,
            "score_gemini": round_score_gem,
        }
        rows.append(row)

        # write immediately
        csv_writer.writerow(row)
        csv_f.flush()

        print(f"[Pairwise] round {i+1}: A_is_ft={a_is_ft}, ft_round={round_score_ft}, gem_round={round_score_gem}")
        time.sleep(1)

    totals = {"score_finetuned": score_ft, "score_gemini": score_gemini}
    csv_f.write("\n")
    csv_f.write("Totals:\n")
    for k, v in totals.items():
        csv_f.write(f"{k},{v}\n")
    csv_f.close()
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
