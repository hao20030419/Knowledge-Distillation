import os
import json
import csv
from multiprocessing import Process, Queue

from GPTagent.agent.generator import generate_raw_question, random_topic
from GPTagent.agent.beautifier import beautify
from GPTagent.agent.reviewer import review
from GPTagent.agent.fixer import fix_format

# 正確定位 GPTagent/results 目錄
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # GPTagent/
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


def process_question(topic):
    raw = generate_raw_question(topic)
    pretty = beautify(raw)
    reviewed = review(pretty)
    fixed = fix_format(reviewed["final_question"])

    lines = fixed.splitlines()
    ans = next((l for l in lines if l.startswith("答案")), "答案：N/A")
    exp = next((l for l in lines if l.startswith("解析")), "解析：N/A")
    
    return {
        "topic": topic,
        "original": raw,
        "final": fixed,
        "answer": ans.replace("答案：", "").strip(),
        "explain": exp.replace("解析：", "").strip(),
        "decision": reviewed["decision"],
        "reason": reviewed["reason"],
        "review_json": json.dumps(reviewed, ensure_ascii=False)
    }


def worker(task_q, result_q):
    while True:
        topic = task_q.get()
        if topic == "__END__":
            break
        result_q.put(process_question(topic))


def writer(result_q, total, jsonl_path, csv_path):
    done = 0

    with open(jsonl_path, "w", encoding="utf-8") as fj, \
         open(csv_path, "w", newline="", encoding="utf-8-sig") as fc:

        w = csv.writer(fc)
        w.writerow(["Topic", "Decision", "Reason",
                    "Final", "Answer", "Explain",
                    "Original", "Review JSON"])

        while done < total:
            r = result_q.get()
            done += 1

            fj.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "MCQ generation"},
                    {"role": "assistant", "content": r["final"]}
                ]
            }, ensure_ascii=False) + "\n")

            w.writerow([
                r["topic"], r["decision"], r["reason"],
                r["final"], r["answer"], r["explain"],
                r["original"], r["review_json"]
            ])

            print(f"[Writer] 已完成 {done}/{total}")


def generate_dataset(total=20, workers=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_q = Queue()
    result_q = Queue()

    for _ in range(total):
        task_q.put(random_topic())
    for _ in range(workers):
        task_q.put("__END__")

    ps = []
    for _ in range(workers):
        p = Process(target=worker, args=(task_q, result_q))
        p.start()
        ps.append(p)

    jsonl_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    csv_path = os.path.join(OUTPUT_DIR, "review.csv")

    wp = Process(target=writer, args=(result_q, total, jsonl_path, csv_path))
    wp.start()

    for p in ps:
        p.join()
    wp.join()

    print("=== 產生完成 ===")
