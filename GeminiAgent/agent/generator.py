import os
import random
from google import genai
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

QUESTION_PROMPT = """
你是一位資工領域的出題專家。請根據主題編寫一道單選題（四選一），並提供題目與四個選項。
使用繁體中文。

格式：
題目：xxxx
(A) xxx
(B) xxx
(C) xxx
(D) xxx
答案：X
解析：xxxx
"""

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


def call_gemini(prompt):
    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt
    )
    return resp.text.strip() if resp.text else ""


def generate_raw_question(topic):
    msgs = f"{QUESTION_PROMPT}\n請根據主題「{topic}」出一題。"
    out = call_gemini(msgs)

    if not out:
        out = f"題目（Gemini 回傳空白）請出與 {topic} 有關的題目。"

    return out


def random_topic():
    return random.choice(CS_TOPICS)