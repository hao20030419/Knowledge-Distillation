import json
import os
import random
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# === 正確 results 資料夾 ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # → GeminiAgent/
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# === 嚴格審題器 Prompt ===
FINAL_QA_PROMPT = """
你是一位極度嚴格的題目檢查員。

請檢查下面單選題是否完整、清楚、答案唯一、解析合理。

請回覆 JSON：
{
  "keep": true 或 false,
  "reason": "原因"
}
"""

# ---------------------------------------------------
# Prompt 模板：讓 question 看起來更自然
# ---------------------------------------------------
PROMPT_TEMPLATES = [
    "請幫我出一題{subject}的單選題",
    "我想練習{subject}，請給我一題四選一題目",
    "可以出一題與{subject}相關的 MCQ 題目嗎？",
    "請生成一題{subject}領域的選擇題（四選一）",
    "請提供一題{subject}的考試題目（四選一）",
]

def random_prompt(subject: str):
    return random.choice(PROMPT_TEMPLATES).format(subject=subject)


# ---------------------------------------------------
# Gemini 審題
# ---------------------------------------------------
def llm_check(text):
    try:
        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[
                {"role": "system", "content": FINAL_QA_PROMPT},
                {"role": "user", "content": text}
            ]
        )
        txt = resp.text
        s = txt[txt.find("{"): txt.rfind("}") + 1]
        data = json.loads(s)
        return data.get("keep", False), data.get("reason", "")
    except Exception:
        return False, "審查解析失敗"


# ---------------------------------------------------
# topic → 資訊工程{科目}
# 例：
#   資料結構 - 陣列(Array) → 資訊工程資料結構
# ---------------------------------------------------
def extract_subject(topic: str):
    if " - " in topic:
        field = topic.split(" - ")[0].strip()
        return f"資訊工程{field}"
    return "資訊工程"


# ---------------------------------------------------
# 主清洗流程
# ---------------------------------------------------
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
            full = data.get("question", "") or data["question"]

            # Gemini 查核題目品質
            ok, reason = llm_check(full)

            if not ok:
                drop += 1
                print(f"[DROP] {reason}")
                fdrop.write(json.dumps({
                    "reason": reason,
                    "content": full
                }, ensure_ascii=False) + "\n")
                continue

            keep += 1
            print(f"[KEEP] {reason}")

            # ---------------------------------------------------
            # 移除「答案：」與「解析：」
            # ---------------------------------------------------
            lines = full.splitlines()
            llmans = "\n".join([
                l for l in lines
                if not l.strip().startswith("答案")
                and not l.strip().startswith("解析")
            ]).strip()

            # ---------------------------------------------------
            # 產生 prompt-like 的 question
            # ---------------------------------------------------
            subject = extract_subject(topic)
            question_text = random_prompt(subject)

            fout.write(json.dumps({
                "question": question_text,
                "LLMans": llmans
            }, ensure_ascii=False) + "\n")

    print("\n=== 清洗完成 ===")
    print(f"✔ 保留：{keep}")
    print(f"✖ 移除：{drop}")
    print(f"📄 輸出：{out}")
