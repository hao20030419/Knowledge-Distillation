import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

MINI_KEY = os.getenv("REVIEW_MODEL_API_KEY")
mini = OpenAI(api_key=MINI_KEY)

# -------------------------
# Final QA Agent Prompt
# -------------------------
FINAL_QA_PROMPT = """
你是一位極度嚴格的題目檢查員。

請檢查下面「單選題」是否合格：

必須符合：
- 必須是單選題
- 必須只有四個選項 (A)(B)(C)(D)
- 必須有一個且只有一個正確答案
- 必須有解析
- 必須為電腦科學領域（資料結構 / 演算法 / OS / DB / 網路）
- 必須與主題高度相關
- 不得出現非 CS 的題目（如化學、水、植物）
- 不得出現多題混在一起
- 不得出現奇怪格式（如編號 1.2.3.、文章、問答題）
- 不得只有題目沒有選項
- 不得答案錯誤或不合理

請回覆純 JSON：

{
  "keep": true 或 false,
  "reason": "簡短說明原因"
}
"""

def call_mini(messages, retry=3):
    for _ in range(retry):
        try:
            resp = mini.responses.create(
                model="gpt-4o-mini",
                input=messages,
                max_output_tokens=300
            )
            return resp.output_text
        except:
            pass
    return ""

def is_good_mcq(question_text):
    """Return True/False based on final QA model judgement."""
    msgs = [
        {"role": "system", "content": FINAL_QA_PROMPT},
        {"role": "user", "content": question_text}
    ]

    out = call_mini(msgs)
    if not out.strip():
        return False

    try:
        s = out[out.find("{"): out.rfind("}") + 1]
        data = json.loads(s)
        return data.get("keep", False), data.get("reason", "")
    except:
        return False, "JSON parse failed"

# -------------------------
# 清洗流程
# -------------------------

def clean_dataset(
    source="dataset.jsonl",
    output="clean_dataset.jsonl",
    removed="removed.jsonl"
):
    keep_count = 0
    drop_count = 0

    with open(source, "r", encoding="utf-8") as f_in, \
         open(output, "w", encoding="utf-8") as f_out, \
         open(removed, "w", encoding="utf-8") as f_bad:

        for line in f_in:
            try:
                data = json.loads(line)
                mcq = data["messages"][1]["content"]  # assistant 的最終題目
            except:
                continue

            ok, reason = is_good_mcq(mcq)

            if ok:
                f_out.write(line)
                keep_count += 1
                print(f"[KEEP] {reason}")
            else:
                f_bad.write(json.dumps({
                    "reason": reason,
                    "question": mcq
                }, ensure_ascii=False) + "\n")
                drop_count += 1
                print(f"[DROP] {reason}")

    print("\n=== 清洗完成 ===")
    print(f"保留：{keep_count}")
    print(f"移除：{drop_count}")
    print(f"輸出檔：{output}")
    print(f"剔除檔：{removed}")

if __name__ == "__main__":
    clean_dataset()
