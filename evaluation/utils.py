import os
import re
import csv
import json
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# reuse existing Gemini helper
from GeminiAgent.agent.llm_utils import generate_content_with_tokens
from GeminiAgent.agent.generator import random_topic


def load_finetuned_model(model_dir: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
    """Load a fine-tuned model directory. Returns (model, tokenizer, gen_pipeline).
    
    Args:
        model_dir: Path to the model or adapter.
        load_in_8bit: Use 8-bit quantization.
        load_in_4bit: Use 4-bit quantization.
    """
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|endoftext|>"})

    # Prepare quantization config
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

    # Try loading with device_map auto
    device_map = "auto" if torch.cuda.is_available() else None

    # Helper to load base
    def _load_base(path):
        return AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            quantization_config=bn_config,
            trust_remote_code=True
        )

    try:
        # First attempt: assume model_dir is a full model or base model
        # If it's an adapter, this might load the base if specified in config, or fail.
        # However, for PEFT adapters, we usually need to load base first separately if we knew what it was.
        # But here we try to be generic. 
        # Strategy: try loading as AutoModel. If it's an adapter, this often works IF adapter_config.json has 'base_model_name_or_path' AND that path is valid locally/remote.
        # BUT capturing the "it is an adapter" case is tricky with just from_pretrained().
        
        # Let's try loading as base first.
        base = _load_base(model_dir)
        
        # If successful, check if we need to wrap it with Peft (e.g. if model_dir ITSELF was the adapter, 
        # but AutoModel loaded the underlying base).
        # Actually AutoModel.from_pretrained(adapter_dir) usually loads the BASE model logic if peft is installed? 
        # No, usually it loads the Config and weights from base_model_name_or_path.
        
        # Let's try to explicitly apply PEFT if model_dir contains adapter_config.json
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            try:
                model = PeftModel.from_pretrained(base, model_dir)
            except Exception:
                model = base
        else:
            model = base

    except Exception:
        # Fallback? If simple load failed, maybe it's because of some structure issue.
        # We re-raise or try a simpler load without quantization?
        # But if quantization was requested, we shouldn't fallback to full precision silently (OOM risk).
        # Let's try to load without specific device map if that was the issue? (Rare)
        # Instead, assume failure.
        print(f"Warning: Failed to load {model_dir} with quantization. Trying fallback...")
        base = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
        model = base
    
    # When the model is loaded with `accelerate` it manages device placement,
    # so avoid passing the `device` argument to `pipeline()` which would try
    # to move the model again and raise an error.
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, gen


def gen_from_finetuned(gen_pipeline, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
    # defaults
    generation_kwargs = {"do_sample": False}
    generation_kwargs.update(kwargs)

    # If sampling parameters are present, force do_sample=True unless strictly disabled
    if any(k in generation_kwargs for k in ["temperature", "top_p", "top_k"]):
        if generation_kwargs.get("temperature", 1.0) == 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True

    out = gen_pipeline(prompt, max_new_tokens=max_new_tokens, **generation_kwargs)
    if isinstance(out, list) and out:
        text = out[0].get("generated_text") or out[0].get("text") or ""
    else:
        text = ""
    return text.strip()


def gen_from_gemini(prompt: str, model: str = "gemini-3-pro-preview") -> Tuple[str, int, int]:
    text, out_tokens, in_tokens = generate_content_with_tokens(model, prompt)
    return text.strip(), out_tokens, in_tokens


def gen_from_gpt(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 5000) -> str:
    """Call OpenAI Chat API (if available) and return the assistant text.

    Returns the generated text. If `openai` is not installed or API key missing,
    raises RuntimeError with an explanatory message.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package is not installed. Install with `pip install openai` to use GPT judge.")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY") or os.getenv("QUESTION_MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found in OPENAI_API_KEY, GPT_API_KEY or QUESTION_MODEL_API_KEY environment variable.")

    # Support both new openai.OpenAI client and legacy openai.ChatCompletion
    def _extract_text(resp):
        try:
            if isinstance(resp, dict):
                if "choices" in resp and resp["choices"]:
                    first = resp["choices"][0]
                    if isinstance(first, dict):
                        return first.get("message", {}).get("content") or first.get("text") or ""
                if "output_text" in resp:
                    return resp.get("output_text", "")
                if "output" in resp and resp["output"]:
                    out0 = resp["output"][0]
                    if isinstance(out0, dict):
                        cont = out0.get("content") or out0.get("text")
                        if isinstance(cont, list) and cont:
                            texts = []
                            for c in cont:
                                if isinstance(c, dict) and "text" in c:
                                    texts.append(c["text"])
                                elif isinstance(c, str):
                                    texts.append(c)
                            return "\n".join(texts)
                    return str(out0)
                return ""

            # object-like
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                try:
                    return first.message.content
                except Exception:
                    try:
                        return first.get("message", {}).get("content", "")
                    except Exception:
                        return str(first)

            output = getattr(resp, "output", None)
            if output:
                first = output[0]
                cont = getattr(first, "content", None)
                if isinstance(cont, list) and cont:
                    texts = []
                    for c in cont:
                        if isinstance(c, dict) and "text" in c:
                            texts.append(c["text"])
                        elif isinstance(c, str):
                            texts.append(c)
                    return "\n".join(texts)
                try:
                    return str(first)
                except Exception:
                    return ""
        except Exception:
            return ""

    try:
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            # prefer chat.completions with max_completion_tokens for newer models
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                )
            except Exception as e_chat:
                # try the Responses API as a fallback (different parameter names)
                try:
                    resp = client.responses.create(
                        model=model,
                        input=[{"role": "user", "content": prompt}],
                        max_output_tokens=max_tokens,
                    )
                except Exception as e_resp:
                    raise RuntimeError(f"OpenAI API chat and responses attempts failed: {e_chat}; {e_resp}")

            text = _extract_text(resp)
            return text.strip()

        # legacy SDK fallback
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        choices = resp.get("choices") or []
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        else:
            text = ""
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


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
