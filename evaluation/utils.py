import os
import re
import csv
import json
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# reuse existing Gemini helper
from GeminiAgent.agent.llm_utils import generate_content_with_tokens
from GeminiAgent.agent.generator import random_topic


def load_finetuned_model(model_dir: str):
    """Load a fine-tuned model directory. Returns (model, tokenizer, gen_pipeline).

    This tries to wrap the base model with PEFT adapter if present.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|endoftext|>"})

    # Try loading with device_map auto
    device_map = "auto" if torch.cuda.is_available() else None
    try:
        base = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
        # Try to wrap with PEFT adapter if present
        try:
            model = PeftModel.from_pretrained(base, model_dir)
        except Exception:
            model = base
    except Exception:
        # fallback to direct load
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    
    # When the model is loaded with `accelerate` it manages device placement,
    # so avoid passing the `device` argument to `pipeline()` which would try
    # to move the model again and raise an error.
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, gen


def gen_from_finetuned(gen_pipeline, prompt: str, max_new_tokens: int = 256) -> str:
    out = gen_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    if isinstance(out, list) and out:
        text = out[0].get("generated_text") or out[0].get("text") or ""
    else:
        text = ""
    return text.strip()


def gen_from_gemini(prompt: str, model: str = "gemini-3-pro-preview") -> Tuple[str, int, int]:
    text, out_tokens, in_tokens = generate_content_with_tokens(model, prompt)
    return text.strip(), out_tokens, in_tokens


def parse_score_from_text(text: str) -> int:
    """Extract an integer score 1-10 from text. Returns -1 if not found."""
    # Prefer explicit 'Score' labels (e.g. 'Score: 8' or 'Score：8').
    # Match both ASCII and full-width colon/dash separators and allow whitespace.
    m = re.search(r"Score\s*[:：-]?\s*(\d{1,2})", text, flags=re.IGNORECASE)
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return v
        except Exception:
            return -1

    # Fallback: look for the pattern 'Score ... X' (Score before the number)
    m2 = re.search(r"Score[^\d]{0,6}(\d{1,2})", text, flags=re.IGNORECASE)
    if m2:
        try:
            v = int(m2.group(1))
            if 1 <= v <= 10:
                return v
        except Exception:
            return -1

    # If no explicit Score label is found, do not pick arbitrary standalone numbers.
    return -1


def parse_winner_from_text(text: str) -> str:
    """Parse winner choice from text: returns 'A' or 'B' or '' if not found."""
    txt = text.upper()
    if "WINNER: A" in txt or "CHOICE: A" in txt or "A)" in txt and "B)" not in txt:
        return "A"
    if "WINNER: B" in txt or "CHOICE: B" in txt or "B)" in txt and "A)" not in txt:
        return "B"
    # look for standalone A or B
    if re.search(r"\bA\b", txt) and not re.search(r"\bB\b", txt):
        return "A"
    if re.search(r"\bB\b", txt) and not re.search(r"\bA\b", txt):
        return "B"
    # last resort: look for words 'A' or 'B' near 'winner' or 'better'
    m = re.search(r"(A|B)[^\n\r]{0,30}(?:is better|is superior|winner|prefer)", txt)
    if m:
        return m.group(1)
    return ""


def save_rounds_csv(path: str, rows: list, totals: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        # write totals as final lines
        f.write("\n")
        f.write("Totals:\n")
        for k, v in totals.items():
            f.write(f"{k},{v}\n")
