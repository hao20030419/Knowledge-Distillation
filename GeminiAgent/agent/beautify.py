import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BEAUTIFY_PROMPT = """
請將以下題目排版乾淨，移除奇怪符號，但不要修改內容：
"""


def beautify(text):
    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=f"{BEAUTIFY_PROMPT}\n{text}"
    )
    return resp.text.strip()