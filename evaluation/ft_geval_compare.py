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
    total_before = 0
    total_after = 0

    # Prepare CSV for incremental writes
    fieldnames = [
        "round",
        "topic",
        "A_is_after",
        "question_A",
        "question_B",
        "score_A",
        "score_B",
        "score_before",
        "score_after",
        "judge_raw",
    ]
    csv_f = open(args.output_csv, "w", encoding="utf-8-sig", newline="")
    csv_writer = __import__("csv").DictWriter(csv_f, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_f.flush()

    for i in range(args.repeats):
        topic = random_topic()
        prompt_pool = [
        "請幫我出一題有關 {topic} 的單選題(四個選項)",
        "你是一位大學的資工老師，幫我針對 {topic} 出一個考試用的單選題（四選一）",
        "我想要一題關於 {topic} 的單選題(四個選項)。",
        "以考古題風格出一題關於 {topic} 的單選題。",
        "請幫我設計一道有關 {topic} 的單選題（四個選項）。",
        "幫我生成一題選擇題，主題是 {topic}。",
        "針對 {topic} 這個主題，請幫我出個單選題，要有四個選項。",
        "能不能給我一題 {topic} 的單選題？記得要是四選一的形式。",
        "請以 {topic} 為題，做一題有四個選項的單選題給我。",
        "幫忙出一道 {topic} 的單選題，選項要給四個。",
        "假設你是資工系教授，請出一個關於 {topic} 的期考單選題（四選一）。",
        "請扮演大學資工老師，幫我設計一題 {topic} 的四選一考試題。",
        "以資工系老師的角度，出一題 {topic} 的考試用單選題，要有四個選項。",
        "模擬資工教授出題，針對 {topic} 做一個四選一的單選考題。",
        "給我一題 {topic} 的單選題，形式要是四個選項。",
        "我想練習一題 {topic} 的單選題，請提供四個選項。",
        "麻煩給我一道 {topic} 的四選一單選題。",
        "請提供一題關於 {topic} 的單選題，並附上四個選項。",
        "請模仿歷屆考題的風格，出一道 {topic} 的單選題。",
        "用考古題的那種感覺，幫我出一題 {topic} 的單選題。",
        "請模擬考古題的語氣，針對 {topic} 設計一題單選題。",
        "幫我出一題 {topic} 的單選題，風格要像考古題一樣。",
        "請協助設計一題 {topic} 的四選一單選題。",
        "幫我構思一道關於 {topic} 的單選題，要有四個選項。",
        "請針對 {topic} 製作一題單選題（四個選項）。",
        "設計一個 {topic} 的單選題給我，選項要給四個。",
        "請產生一題以 {topic} 為主題的選擇題。",
        "主題設定為 {topic}，幫我生成一道選擇題。",
        "針對 {topic} 主題，自動生成一題選擇題。",
        "做一題選擇題給我，題目要跟 {topic} 有關。",
        "你是一位出題風格嚴謹的魔鬼考官。請針對 {topic} 設計一道高難度的單選題，其中必須包含一個極具誘惑性的「陷阱選項」，以測試考生是否真正融會貫通。",
        "請扮演一位熱心的助教，為了幫助學生複習 {topic}，設計一題觀念型的四選一單選題，並在答案後方附上詳盡的解析，解釋為何該選項正確而其他選項錯誤。",
        "你是該領域的資深顧問，請以「實際應用情境」為背景，出一道關於 {topic} 的情境式單選題，避免單純考記憶背誦，而是著重於解決問題的能力。",
        "假設你正在進行技術面試，請針對 {topic} 出一道用來評估求職者專業程度的單選題，題目敘述要簡潔專業，選項設計要能區分出初學者與專家的差別。"
        ]
        prompt = random.choice(prompt_pool).format(topic=topic)

        q_before = gen_from_finetuned(gen_before, prompt, **gen_kwargs)
        q_after = gen_from_finetuned(gen_after, prompt, **gen_kwargs)

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
            "請先給出每題的評分理由，再在各自理由最後一行回報 'Score: X'（X 為 1-10 的整數）。只要數字即可作為分數行的結尾。"
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

        row = {
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
        }
        rows.append(row)

        # write immediately
        csv_writer.writerow(row)
        csv_f.flush()

        print(f"[FT G-Eval] round {i+1}: A_is_after={a_is_after}, before={score_before}, after={score_after}")
        time.sleep(1)

    totals = {"total_before": total_before, "total_after": total_after}
    csv_f.write("\n")
    csv_f.write("Totals:\n")
    for k, v in totals.items():
        csv_f.write(f"{k},{v}\n")
    csv_f.close()
    print("Saved results to", args.output_csv)


if __name__ == "__main__":
    main()
