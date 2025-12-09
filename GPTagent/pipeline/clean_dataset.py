import json
import os
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MINI_KEY = os.getenv("REVIEW_MODEL_API_KEY")
mini = OpenAI(api_key=MINI_KEY)

# === results 路徑 ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # → GPTagent/
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# === 嚴格題目審查 Prompt ===
FINAL_QA_PROMPT = """
你是一位極度嚴格的題目檢查員。

請檢查下面的單選題是否完整、正確、格式良好。

回覆 JSON：
{
  "keep": true 或 false,
  "reason": "原因"
}
"""

# ---------------------------------------------------------------------
# 隨機 prompt 模板：讓 question 變自然
# ---------------------------------------------------------------------
PROMPT_TEMPLATES = [
    "請幫我出{subject}的單選題",
    "我希望你根據{subject}，請給我一題四選一題目",
    "請生成與{subject}相關的題目",
    "可以出{subject}相關的考題嗎？四選一即可",
    "請給我一題{subject}領域的選擇題（四選一）",
]


def random_prompt(subject: str):
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(subject=subject)


# ---------------------------------------------------------------------
# LLM 審查
# ---------------------------------------------------------------------
def llm_check(text):
    try:
        resp = mini.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": FINAL_QA_PROMPT},
                {"role": "user", "content": text}
            ],
            max_output_tokens=300
        ).output_text

        s = resp[resp.find("{"): resp.rfind("}") + 1]
        j = json.loads(s)
        return j.get("keep", False), j.get("reason", "")

    except Exception:
        return False, "審查模型錯誤"


# ---------------------------------------------------------------------
# topic 轉成科目名稱
# 例如：
#   資料結構 - 陣列(Array) → 資訊工程資料結構
# ---------------------------------------------------------------------
def extract_subject(topic: str):
    if " - " in topic:
        field = topic.split(" - ")[0].strip()
        return f"資訊工程{field}"
    return "資訊工程"


# ---------------------------------------------------------------------
# 主流程：清洗 dataset.jsonl → clean_dataset.jsonl
# ---------------------------------------------------------------------
def clean_dataset(source="dataset.jsonl"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    src = os.path.join(OUTPUT_DIR, source)
    out = os.path.join(OUTPUT_DIR, "clean_dataset.jsonl")
    removed = os.path.join(OUTPUT_DIR, "removed.jsonl")

    keep = drop = 0

    print("🧹 Step 2：清洗資料集...")

    with open(src, "r", encoding="utf-8") as fin, \
         open(out, "w", encoding="utf-8") as fout, \
         open(removed, "w", encoding="utf-8") as fdrop:

        for line in fin:
            data = json.loads(line)

            topic = data.get("topic", "")
            full_text = data["messages"][1]["content"]

            # ===== LLM 審查 =====
            ok, reason = llm_check(full_text)

            if not ok:
                drop += 1
                print(f"[DROP] {reason}")
                fdrop.write(json.dumps({
                    "reason": reason,
                    "content": full_text
                }, ensure_ascii=False) + "\n")
                continue

            keep += 1
            print(f"[KEEP] {reason}")

            # ===== 移除答案與解析 =====
            lines = full_text.splitlines()
            llmans = "\n".join([
                l for l in lines
                if not l.startswith("答案") and not l.startswith("解析")
            ]).strip()

            # ===== 產生自然 prompt =====
            subject = extract_subject(topic)
            question_text = random_prompt(subject)

            # ===== 最終輸出 =====
            fout.write(json.dumps({
                "question": question_text,
                "LLMans": llmans
            }, ensure_ascii=False) + "\n")

    print("\n=== 清洗完成 ===")
    print(f"✔ 保留：{keep}")
    print(f"✖ 移除：{drop}")
    print(f"輸出：{out}")