from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

MINI_KEY = os.getenv("REVIEW_MODEL_API_KEY")
mini = OpenAI(api_key=MINI_KEY)

BEAUTIFY_PROMPT = """
你是一位題目優化助手，請將題目整理成乾淨、正確的格式。
不要修改內容，只負責排版與去除奇怪符號。
"""

def beautify(text):
    if not text.strip():
        return text

    msgs = [
        {"role": "system", "content": BEAUTIFY_PROMPT},
        {"role": "user", "content": text}
    ]

    try:
        resp = mini.responses.create(
            model="gpt-4o-mini",
            input=msgs,
            max_output_tokens=300
        )
        return resp.output_text.strip()
    except:
        return text
