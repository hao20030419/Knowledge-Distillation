import os
import sys
import json
import csv
from multiprocessing import Process, Queue
from dotenv import load_dotenv
from dotenv import load_dotenv

# 讓 Windows 下的 multiprocessing 能讀到環境變數
load_dotenv()

# 確保專案根目錄在 sys.path，使得無論在哪個目錄執行都能 import GeminiAgent package
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from GeminiAgent.agent.generator import generate_raw_question, random_topic
from GeminiAgent.agent.comment_agent import comment_question
from GeminiAgent.agent.refine_agent import refine_question
from GeminiAgent.agent.beautify import beautify
from GeminiAgent.agent.reviewer import review_question
import GeminiAgent.agent.generator as generator_module

# GeminiAgent/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


# ----------------------------------------------------
# Worker：Generator → Comment → Beautify → Reviewer
# ----------------------------------------------------
def worker(task_q: Queue, result_q: Queue):
    # 確保每個 worker 子進程都載入最新的 prompts.json（因為子進程有獨立的 module 副本）
    try:
        prompts_path = os.path.join(os.path.dirname(generator_module.__file__), "prompts.json")
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as _f:
                data = json.load(_f)
                if isinstance(data, list) and data:
                    generator_module.PROMPT_TEMPLATES = data
    except Exception as e:
        # 若載入失敗，繼續使用內建範本
        pass

    while True:
        topic = task_q.get()
        if topic == "__END__":
            break

        # 1️⃣ Generator Agent：Gen_LLM 生成原始題目（回傳使用到的人類 prompt 與 LLM 回應）
        used_prompt, raw_question, gen_out_tokens, gen_in_tokens = generate_raw_question(topic)

        # 2️⃣ Comment Agent：comment_LLM 產生修改建議
        comment, comment_out_tokens, comment_in_tokens = comment_question(raw_question)

        # 3️⃣ Comment Agent：refine_LLM 根據建議改寫題目
        refined_question, refine_out_tokens, refine_in_tokens = refine_question(raw_question, comment)

        # 4️⃣ Beautify Agent：beautify_LLM 做排版與符號清理
        pretty_question, beautify_out_tokens, beautify_in_tokens = beautify(refined_question)

        # 5️⃣ Reviewer Agent：keep_or_not_LLM 決定是否保留
        keep, reason, review_out_tokens, review_in_tokens = review_question(pretty_question)

        # dataset.jsonl 必須符合 clean_dataset 的需求
        payload = {
            "topic": topic,
            # 將實際使用到的人類語氣 prompt 存為頂層的 question 欄位
            "question": used_prompt,
            "messages": [
                {"role": "user", "content": used_prompt},
                {"role": "assistant", "content": pretty_question},
            ],
            "keep": keep,
            "reason": reason,
            # 保留中間資訊方便之後 debug / 分析
            "meta": {
                "raw": raw_question,
                "comment": comment,
                "refined": refined_question,
            },
            # token usage stats per stage (approximate if SDK didn't return exact)
            "tokens": {
                "generator_out": gen_out_tokens,
                "generator_in": gen_in_tokens,
                "comment_out": comment_out_tokens,
                "comment_in": comment_in_tokens,
                "refine_out": refine_out_tokens,
                "refine_in": refine_in_tokens,
                "beautify_out": beautify_out_tokens,
                "beautify_in": beautify_in_tokens,
                "review_out": review_out_tokens,
                "review_in": review_in_tokens,
            }
        }

        result_q.put(payload)


# ----------------------------------------------------
# Writer：集中寫入（避免多進程搶寫檔案）
# ----------------------------------------------------
def writer(result_q: Queue, total: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    csv_path = os.path.join(OUTPUT_DIR, "review.csv")
    tokens_csv = os.path.join(OUTPUT_DIR, "tokens.csv")

    done = 0
    with open(jsonl_path, "w", encoding="utf-8") as fj, \
         open(csv_path, "w", newline="", encoding="utf-8-sig") as fc, \
         open(tokens_csv, "w", newline="", encoding="utf-8-sig") as ft:

        w = csv.writer(fc)
        w.writerow(["Topic", "Keep", "Reason", "Question"])

        wt = csv.writer(ft)
        wt.writerow([
            "Topic",
            "Generator_in","Generator_out",
            "Comment_in","Comment_out",
            "Refine_in","Refine_out",
            "Beautify_in","Beautify_out",
            "Review_in","Review_out",
            "TotalInput","TotalOutput","TotalTokens",
            "Keep"
        ])

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

            toks = r.get("tokens", {})
            gen_out = int(toks.get("generator_out") or 0)
            gen_in = int(toks.get("generator_in") or 0)
            com_out = int(toks.get("comment_out") or 0)
            com_in = int(toks.get("comment_in") or 0)
            ref_out = int(toks.get("refine_out") or 0)
            ref_in = int(toks.get("refine_in") or 0)
            bea_out = int(toks.get("beautify_out") or 0)
            bea_in = int(toks.get("beautify_in") or 0)
            rev_out = int(toks.get("review_out") or 0)
            rev_in = int(toks.get("review_in") or 0)

            total_input = sum([gen_in, com_in, ref_in, bea_in, rev_in])
            total_output = sum([gen_out, com_out, ref_out, bea_out, rev_out])
            total_tokens = total_input + total_output

            wt.writerow([
                r.get("topic", ""),
                gen_in, gen_out,
                com_in, com_out,
                ref_in, ref_out,
                bea_in, bea_out,
                rev_in, rev_out,
                total_input, total_output, total_tokens,
                r.get("keep", False),
            ])

            print(f"[Writer] 已完成 {done}/{total}")


# ----------------------------------------------------
# 主程式：產生題目資料集（FT-DataPrep Pipeline）
# ----------------------------------------------------
def generate_dataset(total=1, workers=1):
    task_q = Queue()
    result_q = Queue()

    # 指派 topic 給 Generator
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

    print("=== FT-DataPrep Pipeline：Gemini 資料集生成完成 ===")

if __name__ == "__main__":
    # 允許直接以腳本方式執行：預設產生 20 題、4 個 workers
    # 可依需求改為較小數量做快速驗證
    try:
        # 嘗試從環境變數帶入數量（可選）
        total = int(os.getenv("KD_GEN_TOTAL", "10"))
        workers = int(os.getenv("KD_GEN_WORKERS", "1"))
    except Exception:
        total, workers = 1, 1

    print(f"[Runner] 開始生成資料集：total={total}, workers={workers}")
    generate_dataset(total=total, workers=workers)