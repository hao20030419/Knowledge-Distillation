import os
import random
import json
from google import genai
from dotenv import load_dotenv

"""
Generator Agent
- LLM 名稱：Gen_LLM
- 功能：根據 CS 主題產生原始 MCQ 題目（含答案與解析）
"""

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEN_LLM_NAME = "Gen_LLM"
GEN_MODEL_NAME = "gemini-3-pro-preview"

QUESTION_PROMPT = """
你是一位資工領域的出題專家 (Gen_LLM)。
請根據主題編寫一道單選題（四選一），並提供題目與四個選項、答案與解析。
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

# 人類語氣的 prompt 範本，供隨機挑選以產生多樣化的 QUESTION_PROMPT。
# 這些範本模仿自然講話方式（Alpaca-style augmentation）。
PROMPT_TEMPLATES = [
    "請幫我出一題有關 {topic} 的單選題(四個選項)",
    "你是一位大學的資工老師，幫我針對 {topic} 出一個考試用的單選題（四選一）",
    "我想要一題關於 {topic} 的單選題(四個選項)。",
    "以考古題風格出一題關於 {topic} 的單選題。",
    "請幫我設計一道有關 {topic} 的單選題（四個選項）。",
    "幫我生成一題選擇題，主題是 {topic}。",
]


def build_question_prompt(topic: str) -> str:
    """
    從 `PROMPT_TEMPLATES` 隨機選一個範本並以 topic 格式化，回傳最終要送給 LLM 的 prompt。
    這會用來模擬多樣且自然的人類提問方式（類似 Alpaca 的多樣化 seed）。
    """
    t = random.choice(PROMPT_TEMPLATES).strip()
    return t.format(topic=topic)


# 嘗試載入外部擴增後的 prompts.json（若使用 instruction_generator.py 產生）
try:
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as _f:
            data = json.load(_f)
            if isinstance(data, list) and data:
                PROMPT_TEMPLATES = data
except Exception:
    # 若載入失敗，不影響內建範本
    pass

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


def call_gen_llm(prompt: str) -> str:
    resp = client.models.generate_content(
        model=GEN_MODEL_NAME,
        contents=prompt
    )
    return resp.text.strip() if resp.text else ""


def augment_prompt_templates(target_total: int = 50, per_template: int = 3) -> list:
    """
    使用 LLM 自動擴增 `PROMPT_TEMPLATES`。

    - `target_total`: 擴增後希望的總數量（包含原始範本）。
    - `per_template`: 每個原始範本最多嘗試產生的變體數。

    回傳擴增後的範本清單（list of str）。

    注意：此方法會呼叫 `call_gen_llm`，需設定好 `GEMINI_API_KEY`。
    """
    seeds = PROMPT_TEMPLATES.copy()
    # 快速檢查：如果已經夠多，直接回傳
    if len(seeds) >= target_total:
        return seeds

    # 組裝提示給 LLM：要求輸出 JSON 陣列，保留 {topic} placeholder
    seed_block = "\n".join([f"- {s}" for s in seeds])
    prompt_template = (
        "你是一位提示語（prompt）重寫與變體產生專家，使用繁體中文。\n"
        "請根據下列範本為每一個範本產生最多 {per_template} 個自然口語化但語意等價的變體，\n"
        "保留所有範本中的佔位符 `{topic}`（不要替換），並避免與原始範本重複。\n"
        "請輸出一個 JSON 陣列，內容為字串清單（每個字串為一個變體範本）。\n\n"
        "原始範本：\n"
        f"{seed_block}\n\n"
        "注意：總數量不需超過 target_total；若產生數量過多也沒關係，我會做去重與截斷。"
    )

    # 使用 replace 來替換 {per_template}，避免同時解析到 `{topic}` 造成 KeyError
    prompt = prompt_template.replace("{per_template}", str(per_template))

    resp = call_gen_llm(prompt)
    candidates = []
    if resp:
        # 嘗試解析 JSON
        try:
            j = json.loads(resp)
            if isinstance(j, list):
                candidates = [s.strip() for s in j if isinstance(s, str) and s.strip()]
        except Exception:
            # 如果不是 JSON，嘗試依行拆分（較寬鬆的後備）
            for line in resp.splitlines():
                s = line.strip().lstrip("- ")
                if s:
                    candidates.append(s)

    # 去重並保留原本的 seeds
    seen = set(seeds)
    augmented = seeds.copy()
    for c in candidates:
        if len(augmented) >= target_total:
            break
        # 保留 {topic}，若不包含 placeholder，跳過
        if "{topic}" not in c:
            continue
        if c not in seen:
            augmented.append(c)
            seen.add(c)

    return augmented


def save_prompt_templates(path: str, templates: list):
    """將範本清單存成 JSON 檔案（UTF-8）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 簡單的 CLI：執行此檔會用 LLM 擴增並儲存到 agent/prompts.json
    out_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    print("Start augmenting prompt templates via Gemini...\nThis will call the API (make sure GEMINI_API_KEY is set).")
    new_templates = augment_prompt_templates(target_total=100, per_template=4)
    save_prompt_templates(out_path, new_templates)
    print(f"Saved {len(new_templates)} templates to {out_path}")


def generate_raw_question(topic: str) -> tuple:
    """
    使用 Gen_LLM 依據 topic 產生原始題目（含答案與解析）。

    回傳 (used_prompt, model_output)
    - used_prompt: 實際送給 LLM 的人類語氣 prompt（可存入 dataset 的 question 欄位）
    - model_output: LLM 回傳的題目內容
    """
    # 先建立一個人類語氣的 prompt（多樣化）
    human_prompt = build_question_prompt(topic)

    # 將我們的格式化 QUESTION_PROMPT 與人類 prompt 合併，讓 LLM 既知道要的格式，也感受到自然口語的表述
    msgs = f"{human_prompt}\n\n格式要求：\n{QUESTION_PROMPT}\n請根據主題「{topic}」出一題。"

    out = call_gen_llm(msgs)

    if not out:
        out = f"題目（Gen_LLM 回傳空白）請出與 {topic} 有關的題目。"

    return human_prompt, out


def random_topic() -> str:
    return random.choice(CS_TOPICS)