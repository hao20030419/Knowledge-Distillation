import os
import re
import csv
import json
import torch
import gc
from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

# --- 基礎工具函式 ---

def cleanup_gpu():
    """清理顯存，避免連續載入模型時崩潰"""
    gc.collect()
    torch.cuda.empty_cache()

def load_finetuned_model(model_dir: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
    """
    載入模型，支援 PEFT/LoRA 與 4/8bit 量化。
    """
    model_dir = model_dir.rstrip("/\\")
    is_peft = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    if is_peft:
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model_path = peft_config.base_model_name_or_path
    else:
        base_model_path = model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bn_config = None
    if load_in_4bit:
        bn_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        bn_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bn_config,
        device_map={"": 0}, 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    if is_peft:
        print(f"Loading adapter weights from: {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir)

    # 關鍵修正：return_full_text=False 避免解析到重複的 Prompt
    # 明確指定 device=0
    gen = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=0, 
        return_full_text=False
    )
    return model, tokenizer, gen

def gen_from_finetuned(gen_pipeline, prompt: str, max_new_tokens: int = 1024, **kwargs) -> str:
    """生成回覆並處理重複懲罰"""
    generation_kwargs = {
        "do_sample": True,
        "repetition_penalty": 1.15,
        "temperature": 0.3,
        "top_p": 0.9
    }
    generation_kwargs.update(kwargs)

    out = gen_pipeline(prompt, max_new_tokens=max_new_tokens, **generation_kwargs)
    if isinstance(out, list) and out:
        return out[0].get("generated_text", "").strip()
    return ""

# --- 評審與解析工具 ---

def extract_multi_scores(text: str, num_models: int) -> Dict[int, int]:
    """
    專門為「多模型排名」設計的解析函式。
    由後往前掃描，確保抓到的是 Final Ratings 區塊的最終分數。
    """
    scores = {}
    # 優先切換到 Final Ratings 標籤後的內容
    if "Final Ratings:" in text:
        target_area = text.split("Final Ratings:")[-1]
    else:
        target_area = text

    # 正則：尋找 Model X Score: Y 或 Model X: Y
    # 支援中文冒號、空格與連字號
    pattern = r"Model\s*(\d+)[\s:]*Score?[\s:：-]*(\d+)"
    matches = re.findall(pattern, target_area, re.IGNORECASE)
    
    for m_idx_str, score_str in matches:
        m_idx = int(m_idx_str)
        val = int(score_str)
        if 1 <= m_idx <= num_models:
            scores[m_idx] = val
            
    return scores

# --- 主評分邏輯 (範例) ---

def run_evaluation_round(round_idx, topic, prompt_template, model_dirs, responses_dir):
    """
    執行單輪評分，確保 Topic 變數在生成與評分間嚴格同步。
    """
    # 1. 確定主題
    current_topic = topic
    user_prompt = prompt_template.format(topic=current_topic)
    
    model_names = [os.path.basename(d.rstrip("/\\")) for d in model_dirs]
    
    # 2. 生成內容 (假設已載入並儲存至 responses_dir)
    # ... 此處省略載入模型生成代碼 ...

    # 3. 準備評審
    shuffled_indices = list(range(len(model_names)))
    random.shuffle(shuffled_indices)
    
    judge_content = ""
    idx_to_real_name = {}
    
    for i, s_idx in enumerate(shuffled_indices):
        m_name = model_names[s_idx]
        m_label = i + 1
        idx_to_real_name[m_label] = m_name
        
        fpath = os.path.join(responses_dir, f"round{round_idx}_model_{m_name}.txt")
        with open(fpath, "r", encoding="utf-8") as rf:
            content = rf.read()
        judge_content += f"=== Model {m_label} ===\n{content}\n\n"

    # 4. 構建嚴格的評審 Prompt (加入 Final Ratings 標籤引導)
    judge_prompt = (
        f"你是一位嚴苛的學術評審。請針對主題「{current_topic}」進行排名。\n"
        f"【評分軍規】\n"
        f"1. 強制分差：第一名10分，最後一名1分，分數不得重複。\n"
        f"2. 核心指標：正確性與深度。\n\n"
        f"【待評內容】\n{judge_content}\n"
        f"請先說明理由 (CoT)，最後務必以「Final Ratings:」標籤開頭並按下列格式結尾：\n"
        + "\n".join([f"Model {i+1} Score: X" for i in range(len(model_names))])
    )

    # 5. 呼叫 Gemini 評審
    # 注意：這裡呼叫時傳遞的 judge_prompt 內含 current_topic，絕不重複隨機抽題
    from GeminiAgent.agent.llm_utils import generate_content_with_tokens
    raw_judge_resp, _, _ = generate_content_with_tokens("gemini-3-pro-preview", judge_prompt)
    
    # 6. 解析與映射
    final_scores = extract_multi_scores(raw_judge_resp, len(model_names))
    
    # 對應回原始模型名稱
    round_results = {}
    for m_label, real_name in idx_to_real_name.items():
        score = final_scores.get(m_label, -1)
        round_results[real_name] = score
        
    return round_results, raw_judge_resp