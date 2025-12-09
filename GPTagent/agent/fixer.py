from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

MINI_KEY = os.getenv("REVIEW_MODEL_API_KEY")
mini = OpenAI(api_key=MINI_KEY)

FIX_FORMAT_PROMPT = """
你是一位題目修復助手，請將以下題目強制修復為完整的單選題格式：

題目：xxxx
(A) xxx
(B) xxx
(C) xxx
(D) xxx
答案：X
解析：xxxx
"""

def fix_format(text):
    msgs = [
        {"role": "system", "content": FIX_FORMAT_PROMPT},
        {"role": "user", "content": text}
    ]

    try:
        resp = mini.responses.create(
            model="gpt-4o-mini",
            input=msgs,
            max_output_tokens=500
        )
        return resp.output_text.strip()
    except:
        return text
