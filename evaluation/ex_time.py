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
    parser.add_argument("--after_dir", type=str, required=True, help="path to model after fine-tune")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="evaluation/time_eval_results.csv")
    parser.add_argument("--temperature", type=float, default=0.7, help="generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="generation top_p")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load models in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load models in 4-bit quantization")
    args = parser.parse_args()

    _, _, gen_after = load_finetuned_model(args.after_dir, load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)

    # common generation kwargs
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # Prepare CSV for incremental writes
    fieldnames = [
        "round",
        "topic",
        "question_finetuned",
        "question_gemini",
        "time_finetuned",
        "time_gemini",
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
            "請協助設計一題 {topic} 的四選一單選題。",
            "幫我構思一道關於 {topic} 的單選題，要有四個選項。",
            "請針對 {topic} 製作一題單選題（四個選項）。",
            "設計一個 {topic} 的單選題給我，選項要給四個。",
            "請產生一題以 {topic} 為主題的選擇題。",
            "主題設定為 {topic}，幫我生成一道選擇題。",
            "針對 {topic} 主題，自動生成一題選擇題。",
            "做一題選擇題給我，題目要跟 {topic} 有關。"
        ]

        prompt_template = random.choice(prompt_pool)
        prompt = prompt_template.format(topic = topic)

        q_finetuned = gen_from_finetuned(gen_after, prompt, **gen_kwargs)
        q_germini, _, _ = gen_from_gemini(prompt)

        # test time measurement
        start_time = time.time()
        q_finetuned = gen_from_finetuned(gen_after, prompt, **gen_kwargs)
        end_time = time.time()
        time_finetuned = end_time - start_time
        print(f"Round {i+1} - Finetuned generation time: {time_finetuned:.2f} seconds")

        start_time = time.time()
        q_germini, _, _ = gen_from_gemini(prompt)
        end_time = time.time()
        time_gemini = end_time - start_time
        print(f"Round {i+1} - Gemini generation time: {time_gemini:.2f} seconds")
        csv_writer.writerow({
            "round": i + 1,
            "topic": topic,
            "question_finetuned": q_finetuned,
            "question_gemini": q_germini,
            "time_finetuned": time_finetuned,
            "time_gemini": time_gemini,
        })
        csv_f.flush()

if __name__ == "__main__":
    main()

