from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

MINI_KEY = os.getenv("REVIEW_MODEL_API_KEY")
mini = OpenAI(api_key=MINI_KEY)

REVIEW_PROMPT = """
你是審題專家，請檢查題目是否清楚、選項是否合理、答案是否唯一。
請以純 JSON 格式回覆：

{
  "decision": "accept" 或 "rewrite",
  "reason": "xxx",
  "final_question": "完整題目"
}
"""

def review(text):
    msgs = [
        {"role": "system", "content": REVIEW_PROMPT},
        {"role": "user", "content": text}
    ]

    try:
        resp = mini.responses.create(
            model="gpt-4o-mini",
            input=msgs,
            max_output_tokens=300
        ).output_text
    except:
        return {"decision": "accept", "reason": "error", "final_question": text}

    try:
        s = resp[resp.find("{"): resp.rfind("}") + 1]
        j = json.loads(s)
    except:
        j = {"decision": "accept", "reason": "JSON parse error", "final_question": text}

    if not j.get("final_question"):
        j["final_question"] = text

    return j
