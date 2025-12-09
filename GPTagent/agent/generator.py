import os
import random
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GPT5_KEY = os.getenv("QUESTION_MODEL_API_KEY")
gpt5_client = OpenAI(api_key=GPT5_KEY)


QUESTION_PROMPT = """
你是一位資工領域的出題專家。請根據主題編寫一道單選題（四選一），並提供標準答案與簡短解析。
使用繁體中文。格式如下：

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

def safe_call(messages, retry=3, wait=1.0):
    for _ in range(retry):
        try:
            resp = gpt5_client.responses.create(
                model="gpt-5",
                input=messages,
                max_output_tokens=800
            )
            return resp.output_text
        except:
            time.sleep(wait)
    return ""

def generate_raw_question(topic):
    msgs = [
        {"role": "system", "content": QUESTION_PROMPT},
        {"role": "user", "content": f"請根據主題「{topic}」出一題。"}
    ]
    out = safe_call(msgs, retry=5)
    if not out.strip():
        out = f"題目（gpt5 空白）請出與 {topic} 有關的題目。"
    return out

def random_topic():
    return random.choice(CS_TOPICS)
