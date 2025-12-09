import os
import json
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

REVIEW_PROMPT = """
你是審題專家，請檢查題目是否清楚、選項合理、答案唯一。
請以 JSON 格式回覆：

{
  "keep": true 或 false,
  "reason": "原因"
}
"""


def review_question(text):
    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=f"{REVIEW_PROMPT}\n{text}"
    )

    t = resp.text
    try:
        j = json.loads(t[t.find("{"): t.rfind("}")+1])
        return j.get("keep", False), j.get("reason", "")
    except:
        return False, "JSON parse failed"