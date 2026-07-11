"""
dev_test_client.py — LOCAL TESTING ONLY.

Uses Groq and Gemini as stand-ins for Fireworks so you can sanity-check
routing, prompt templates, and verification logic end-to-end while you
don't yet have real Fireworks credentials.

DO NOT import this from main.py or include it in the submitted container.
Swap back to fireworks_client.py the moment real Fireworks credentials exist.
"""

import requests

# HARDCODE YOUR KEYS HERE
GROQ_API_KEY = ""
GEMINI_API_KEY = ""

# Map your "easy"/"medium"/"hard" tiers to real Groq/Gemini models for testing
DEV_TIER_TO_MODEL = {
    "easy": ("groq", "llama-3.1-8b-instant"),
    "medium": ("groq", "llama-3.3-70b-versatile"),
    "hard": ("gemini", "gemini-2.5-flash"),
}


def call_groq(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_gemini(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    # Safe data extraction to avoid the KeyError
    try:
        candidate = data["candidates"][0]
        # Check if the content was blocked or empty
        if "content" in candidate and "parts" in candidate["content"]:
            return candidate["content"]["parts"][0]["text"]
        else:
            return f"Gemini Empty Response (Finish Reason: {candidate.get('finishReason')})"
    except (KeyError, IndexError):
        return f"Unexpected Gemini JSON Structure: {data}"


def call_dev_model(tier: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Drop-in replacement for call_fireworks() during local dev only."""
    provider, model = DEV_TIER_TO_MODEL[tier]
    if provider == "groq":
        return call_groq(model, system_prompt, user_prompt, max_tokens)
    else:
        return call_gemini(model, system_prompt, user_prompt, max_tokens)


if __name__ == "__main__":
    print("Testing Groq (easy tier)...")
    try:
        print(call_dev_model("easy", "Answer concisely.", "What is 2+2?", 20))
    except Exception as e:
        print(f"❌ Groq failed: {e}")

    print("\nTesting Gemini (hard tier)...")
    try:
        print(call_dev_model("hard", "Answer concisely.", "What is 2+2?", 100))
    except Exception as e:
        print(f"❌ Gemini failed: {e}")