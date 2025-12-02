import os
import json
import csv
import time
import random
import traceback
from multiprocessing import Process, Queue
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 1. 模型初始化
# ==============================
GPT5_KEY = os.getenv("QUESTION_MODEL_API_KEY")
MINI_KEY = os.getenv("REVIEW_MODEL_API_KEY")

gpt5_client = OpenAI(api_key=GPT5_KEY)
mini_client = OpenAI(api_key=MINI_KEY)

CS_TOPICS = [
    "資料結構 - 陣列(Array)", "資料結構 - 連結串列(Linked List)",
    "資料結構 - 堆疊(Stack)", "資料結構 - 佇列(Queue)",
    "資料結構 - Heap", "資料結構 - 二元搜尋樹(BST)",
    "演算法 - Sorting", "演算法 - Searching",
    "演算法 - Dynamic Programming", "演算法 - Greedy",
    "作業系統 - 排程(Scheduling)", "作業系統 - 死結(Deadlock)",
    "計算機網路 - TCP", "計算機網路 - Routing",
    "資料庫 - Transaction", "資料庫 - Index",
]

QUESTION_PROMPT = """
你是一位資工領域的出題專家。請根據主題編寫一道單選題（四選一），並提供標準答案與簡短解析。
使用繁體中文。格式如下：

題目：xxxx
(A) xxx
(B) xxx
(C) xxx
(D) xxx
答案：X
解析：xxxx
"""

BEAUTIFY_PROMPT = """
你是一位題目優化助手，請將題目整理成乾淨、正確的格式。
不要修改內容，只負責排版與去除奇怪符號。
"""

REVIEW_PROMPT = """
你是審題專家，請檢查題目是否清楚、選項是否合理、答案是否唯一。
請以純 JSON 格式回覆：

{
  "decision": "accept" 或 "rewrite",
  "reason": "xxx",
  "final_question": "完整題目（含四選項、答案、解析）"
}
"""

FIX_FORMAT_PROMPT = """
你是一位題目修復助手，請將以下題目強制修復為完整的單選題格式：

要求：
- 必須是單選題（四選項）
- 必須是繁體中文
- 必須包含題目敘述
- 必須包含四個選項 (A)(B)(C)(D)
- 必須包含「答案：X」
- 必須包含「解析：xxxx」

請修復成以下格式：

題目：xxxx
(A) xxx
(B) xxx
(C) xxx
(D) xxx
答案：X
解析：xxxx

只回傳修復後的題目，不允許加其他說明。
"""


# ==============================
# 2. 安全呼叫 LLM
# ==============================
def safe_call(client, model, messages, max_tokens=800, retry=3, wait=1.0):
    for _ in range(retry):
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=max_tokens
            )
            txt = resp.output_text
            if txt and txt.strip():
                return txt
        except Exception as e:
            time.sleep(wait)
    return ""


# ==============================
# 3. 美化
# ==============================
def beautify(text):
    if not text.strip():
        return text
    msgs = [
        {"role": "system", "content": BEAUTIFY_PROMPT},
        {"role": "user", "content": text}
    ]
    out = safe_call(mini_client, "gpt-4o-mini", msgs)
    return out if out.strip() else text


# ==============================
# 4. 審題
# ==============================
def review_question(text):
    msgs = [
        {"role": "system", "content": REVIEW_PROMPT},
        {"role": "user", "content": text}
    ]
    out = safe_call(mini_client, "gpt-4o-mini", msgs)

    # 嘗試解析 JSON
    try:
        s = out[out.find("{"): out.rfind("}") + 1]
        j = json.loads(s)
        if not j.get("final_question"):
            j["final_question"] = text
        return j
    except Exception:
        return {"decision": "accept", "reason": "JSON 解析失敗", "final_question": text}


# ==============================
# 5. 格式強制修復器（Level-3）
# ==============================
def force_fix_format(text):
    msgs = [
        {"role": "system", "content": FIX_FORMAT_PROMPT},
        {"role": "user", "content": text}
    ]
    out = safe_call(mini_client, "gpt-4o-mini", msgs)

    # 若修復失敗 → 再重試 1 次
    if ("答案：" not in out) or ("解析：" not in out):
        out2 = safe_call(mini_client, "gpt-4o-mini", msgs)
        if out2.strip():
            out = out2

    return out


# ==============================
# 6. 完整產生流程
# ==============================
def generate_single(topic):
    try:
        # === (1) GPT-5 出題 ===
        msgs = [
            {"role": "system", "content": QUESTION_PROMPT},
            {"role": "user", "content": f"請根據主題「{topic}」出一題。"}
        ]
        raw = safe_call(gpt5_client, "gpt-5", msgs, retry=5)
        if not raw.strip():
            raw = f"題目（gpt5 空白）請出與 {topic} 有關的題目。"

        # === (2) 美化 ===
        pretty = beautify(raw)

        # === (3) 審題 ===
        reviewed = review_question(pretty)
        final_raw = reviewed.get("final_question") or pretty

        # === (4) Level-3 強制修復 ===
        fixed = force_fix_format(final_raw)

        # === 萃取答案與解析 ===
        lines = fixed.splitlines()
        ans = next((l for l in lines if l.startswith("答案")), "答案：N/A")
        exp = next((l for l in lines if l.startswith("解析")), "解析：N/A")

        return {
            "topic": topic,
            "original": raw,
            "final": fixed,
            "answer": ans.replace("答案：", "").strip(),
            "explain": exp.replace("解析：", "").strip(),
            "decision": reviewed.get("decision", "accept"),
            "reason": reviewed.get("reason", ""),
            "review_json": json.dumps(reviewed, ensure_ascii=False)
        }

    except Exception as e:
        return {
            "topic": topic,
            "original": "",
            "final": "",
            "answer": "N/A",
            "explain": "N/A",
            "decision": "error",
            "reason": str(e),
            "review_json": traceback.format_exc()
        }


# ==============================
# 7. Worker（多程序）
# ==============================
def worker(task_q: Queue, result_q: Queue):
    while True:
        topic = task_q.get()
        if topic == "__END__":
            break
        out = generate_single(topic)
        result_q.put(out)


# ==============================
# 8. Writer（集中寫入）
# ==============================
def writer(result_q: Queue, total, jsonl_path, csv_path):
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

            # JSONL：穩定格式
            fj.write(json.dumps({
                "messages": [
                    {"role": "user", "content": QUESTION_PROMPT},
                    {"role": "assistant", "content": r["final"]}
                ]
            }, ensure_ascii=False) + "\n")

            # CSV
            w.writerow([
                r["topic"], r["decision"], r["reason"],
                r["final"], r["answer"], r["explain"],
                r["original"], r["review_json"]
            ])

            print(f"[Writer] 已完成 {done}/{total}")


# ==============================
# 9. 主程式
# ==============================
def generate_dataset(total=20, workers=4):
    task_q = Queue()
    result_q = Queue()

    # 任務入列
    for _ in range(total):
        task_q.put(random.choice(CS_TOPICS))

    # END signal
    for _ in range(workers):
        task_q.put("__END__")

    # workers
    ps = []
    for _ in range(workers):
        p = Process(target=worker, args=(task_q, result_q))
        p.start()
        ps.append(p)

    # writer
    wp = Process(target=writer,
                 args=(result_q, total, "dataset.jsonl", "review.csv"))
    wp.start()

    # join
    for p in ps:
        p.join()
    wp.join()

    print("\n=== 全部完成 ===")


if __name__ == "__main__":
    generate_dataset(total=20, workers=4)