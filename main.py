"""
main.py — pipeline entrypoint.

Reads /input/tasks.json, classifies each task with the local KNN router,
calls Fireworks (via fireworks_client.py — your friend's piece) with a
category-specific prompt template, verifies the response, and writes
/output/results.json.

This file is the "glue" — it doesn't own routing logic (that's
category_router.py) or the Fireworks call itself (that's fireworks_client.py).
It just wires them together in order.
"""

import json
import os
import sys

from category_router import CategoryRouter, classify_with_fallback, MODEL_TIER

# ---------------------------------------------------------------------------
# Placeholder Fireworks call — replace this import once your friend's
# fireworks_client.py exists. Keeping it inline here means you can test
# the full pipeline end-to-end RIGHT NOW without waiting on that work.
# ---------------------------------------------------------------------------
try:
    from fireworks_client import call_fireworks
except ImportError:
    def call_fireworks(model_id: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Stub — returns a placeholder so the pipeline is testable end-to-end
        before the real Fireworks integration lands."""
        return f"[PLACEHOLDER ANSWER for model={model_id}, category prompt used]"


# ---------------------------------------------------------------------------
# Prompt templates per category: (system_prompt, max_tokens)
# Tune these — this is where real token savings happen.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "factual": ("Answer concisely and accurately in 2-3 sentences.", 450),
    "math": ("Solve step by step, then give the final numeric answer clearly labeled 'Answer:'.", 500),
    "sentiment": ("Classify sentiment as positive, negative, or neutral, then give a one-sentence justification.", 350),
    "summarization": ("Summarize the given text according to the exact length/format constraint specified in the prompt. Do not exceed it.", 450),
    "ner": ("Extract named entities and return ONLY valid JSON in the form "
            '{"entities": [{"text": "...", "type": "PERSON|ORG|LOCATION|DATE"}]}. No extra text.', 550),
    "debugging": ("Identify the bug and provide the corrected code only, with a one-line explanation of the fix.", 800),
    "codegen": ("Write a correct, well-structured function per the spec. Include the function only, no extra commentary.", 900),
    "logic": ("Solve the constraint puzzle. Show your reasoning briefly, then state the final answer clearly.", 800),
}

# ---------------------------------------------------------------------------
# Model tier -> actual Fireworks model ID.
# Fill these in once ALLOWED_MODELS is published on launch day.
# ---------------------------------------------------------------------------
def get_tier_to_model():
    allowed = os.environ.get("ALLOWED_MODELS", "").split(",")
    allowed = [m.strip() for m in allowed if m.strip()]
    if not allowed:
        # fallback for local testing when env var isn't set
        return {"easy": "PLACEHOLDER_EASY_MODEL", "medium": "PLACEHOLDER_MEDIUM_MODEL", "hard": "PLACEHOLDER_HARD_MODEL"}
    # naive default: smallest-listed model = easy, largest-listed = hard, middle = medium
    # replace this with real logic once you know the actual model list/sizes
    return {
        "easy": allowed[0],
        "medium": allowed[len(allowed) // 2],
        "hard": allowed[-1],
    }


# ---------------------------------------------------------------------------
# Verification (stubs — expand these; see previous conversation)
# ---------------------------------------------------------------------------
import ast

def verify_answer(category: str, answer: str) -> bool:
    if category == "ner":
        try:
            json.loads(answer)
            return True
        except json.JSONDecodeError:
            return False
    if category in ("debugging", "codegen"):
        try:
            ast.parse(answer)
            return True
        except SyntaxError:
            return False
    if category == "math":
        return any(char.isdigit() for char in answer)
    return True  # factual, sentiment, summarization, logic: no hard check yet


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"

    with open(input_path, "r") as f:
        tasks = json.load(f)

    router = CategoryRouter(n_neighbors=3)
    tier_to_model = get_tier_to_model()

    results = []
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]

        category, _source = classify_with_fallback(router, prompt)
        tier = MODEL_TIER[category]
        model_id = tier_to_model[tier]
        system_prompt, max_tokens = PROMPT_TEMPLATES[category]

        answer = call_fireworks(model_id, system_prompt, prompt, max_tokens)

        if not verify_answer(category, answer):
            # escalate once to the "hard" tier model with same prompt
            escalated_model = tier_to_model["hard"]
            answer = call_fireworks(escalated_model, system_prompt, prompt, max_tokens)

        results.append({"task_id": task_id, "answer": answer})

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)