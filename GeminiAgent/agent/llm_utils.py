import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_content_with_tokens(model: str, contents: str):
    """Call Gemini generate_content and try to extract token usage.

    Returns: (text: str, out_tokens: int, in_tokens: int)
    - out_tokens: token usage for the model's output (best-effort from SDK or fallback estimate)
    - in_tokens: estimated token count for the input `contents` (simple whitespace-based estimate)

    If the SDK response doesn't expose token usage, out_tokens falls back to a simple
    word-count estimate of the output text.
    """
    # estimate input tokens (simple heuristic)
    in_tokens = len(contents.split())

    resp = client.models.generate_content(model=model, contents=contents)
    text = resp.text.strip() if resp.text else ""

    out_tokens = None
    try:
        # Try common patterns used by SDK responses
        if hasattr(resp, "tokenUsage"):
            tu = getattr(resp, "tokenUsage")
            if isinstance(tu, dict):
                out_tokens = tu.get("totalTokens") or tu.get("total_tokens") or tu.get("total")
            elif hasattr(tu, "totalTokens"):
                out_tokens = tu.totalTokens

        if out_tokens is None and hasattr(resp, "usage"):
            u = getattr(resp, "usage")
            if isinstance(u, dict):
                out_tokens = u.get("total_tokens") or u.get("completion_tokens") or u.get("total")

        if out_tokens is None and hasattr(resp, "metadata"):
            meta = getattr(resp, "metadata")
            if isinstance(meta, dict):
                out_tokens = meta.get("token_count") or meta.get("tokens") or meta.get("tokenCount")
    except Exception:
        out_tokens = None

    # Fallback: estimate by splitting on whitespace
    if out_tokens is None:
        out_tokens = len(text.split())

    try:
        out_tokens = int(out_tokens)
    except Exception:
        out_tokens = len(text.split())

    return text, out_tokens, in_tokens
