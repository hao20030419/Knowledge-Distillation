import os
import json
import csv
import random
from multiprocessing import Process, Queue
from dotenv import load_dotenv

# 讓 Windows 下的 multiprocessing 能讀到環境變數
load_dotenv()

from GeminiAgent.agent.generator import generate_raw_question, random_topic
from GeminiAgent.agent.beautify import beautify
from GeminiAgent.agent.reviewer import review_question


BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # GeminiAgent/
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


# ----------------------------------------------------
# Worker：產生 + 美化 + 審題
# ----------------------------------------------------
def worker(task_q, result_q):
    while True:
        topic = task_q.get()
        if topic == "__END__":
            break

        raw = generate_raw_question(topic)
        pretty = beautify(raw)
        keep, reason = review_question(pretty)

        # dataset.jsonl 必須符合 clean_dataset 要求
        payload = {
            "topic": topic,
            "messages": [
                {"role": "user", "content": "MCQ generation"},
                {"role": "assistant", "content": pretty}
            ],
            "keep": keep,
            "reason": reason
        }

        result_q.put(payload)


# ----------------------------------------------------
# Writer：集中寫入（避免多進程搶寫檔案）
# ----------------------------------------------------
def writer(result_q, total):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    csv_path = os.path.join(OUTPUT_DIR, "review.csv")

    done = 0
    with open(jsonl_path, "w", encoding="utf-8") as fj, \
         open(csv_path, "w", newline="", encoding="utf-8-sig") as fc:

        w = csv.writer(fc)
        w.writerow(["Topic", "Keep", "Reason", "Question"])

        while done < total:
            r = result_q.get()
            done += 1

            fj.write(json.dumps(r, ensure_ascii=False) + "\n")

            w.writerow([
                r["topic"],
                r["keep"],
                r["reason"],
                r["messages"][1]["content"],
            ])

            print(f"[Writer] 已完成 {done}/{total}")


# ----------------------------------------------------
# 主程式：產生題目資料集
# ----------------------------------------------------
def generate_dataset(total=20, workers=4):
    task_q = Queue()
    result_q = Queue()

    # assign topics
    for _ in range(total):
        task_q.put(random_topic())

    # 給 worker 結束 signal
    for _ in range(workers):
        task_q.put("__END__")

    # 啟動 worker
    ps = []
    for _ in range(workers):
        p = Process(target=worker, args=(task_q, result_q))
        p.start()
        ps.append(p)

    # 啟動 writer
    wp = Process(target=writer, args=(result_q, total))
    wp.start()

    # 等 worker 結束
    for p in ps:
        p.join()

    # 等 writer 結束
    wp.join()

    print("=== Gemini 資料集生成完成 ===")