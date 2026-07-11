"""
fireworks_client.py — wraps calls to the Fireworks AI API.

ALL inference must go through FIREWORKS_BASE_URL using a model from
ALLOWED_MODELS. Never hardcode API keys, base URLs, or model IDs —
the harness injects them via environment variables at runtime.
"""

import os
import requests
from dotenv import load_dotenv
load_dotenv()

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]


def call_fireworks(model_id: str, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    if not FIREWORKS_API_KEY or not FIREWORKS_BASE_URL:
        raise RuntimeError(
            "FIREWORKS_API_KEY / FIREWORKS_BASE_URL not set. "
            "These must come from the harness environment, not a .env file, "
            "in the final submitted container."
        )

    if ALLOWED_MODELS and model_id not in ALLOWED_MODELS:
        raise ValueError(
            f"Model '{model_id}' is not in ALLOWED_MODELS ({ALLOWED_MODELS}). "
            "Calling a disallowed model invalidates the submission — refusing to proceed."
        )

    url = f"{FIREWORKS_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Fireworks response shape: {data}") from e

    content = message.get("content")
    finish_reason = choice.get("finish_reason")

    if not content:
        reasoning = message.get("reasoning_content", "")
        usage = data.get("usage", {})
        raise ValueError(
            f"Model '{model_id}' returned no content (finish_reason={finish_reason}, "
            f"completion_tokens={usage.get('completion_tokens')}). "
            f"Likely ran out of tokens mid-reasoning — increase max_tokens. "
            f"Reasoning so far: {reasoning[:200]}"
        )

    return content


if __name__ == "__main__":
    if not ALLOWED_MODELS:
        print("ALLOWED_MODELS not set — can't run a live test. Set env vars first.")
    else:
        test_model = ALLOWED_MODELS[0]
        print(f"Testing call_fireworks with model: {test_model}")
        result = call_fireworks(
            model_id=test_model,
            system_prompt="Answer concisely.",
            user_prompt="What is 2 + 2?",
            max_tokens=500,
        )
        print(f"Response: {result}")