from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from llama_cpp import Llama
from openai import AsyncOpenAI

from routezero.config import Settings


class LLMClient(Protocol):
    """Protocol for all LLM clients."""
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    latency_ms: float
    model_name: str


class LocalQwenClient:
    """Client for local Qwen 2.5 1.5B (GGUF, CPU inference via llama.cpp).

    Sized to fit the 4GB RAM / 2 vCPU grading environment. Local inference
    counts toward accuracy but records zero tokens toward the score.
    """

    def __init__(self, settings: Settings) -> None:
        self.model_path = "./model_cache/local_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=2,
            verbose=False,
        )

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start = time.monotonic()

        # llama_cpp is synchronous; run in a thread so we don't block the
        # asyncio event loop (other async work, e.g. metrics logging, can
        # still proceed).
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            ),
        )

        latency_ms = (time.monotonic() - start) * 1000
        logging.debug("RAW LOCAL RESPONSE: %s", output)

        if not output.get("choices"):
            raise RuntimeError(f"Local model returned no choices. Full response: {output}")

        choice = output["choices"][0]
        usage = output.get("usage", {})
        return LLMResponse(
            text=choice["message"]["content"] or "",
            tokens_used=usage.get("total_tokens", 0),
            latency_ms=round(latency_ms, 2),
            model_name="qwen2.5-1.5b-instruct-local",
        )

    async def health_check(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                ),
            )
            return True
        except Exception:
            return False


class FireworksRemoteClient:
    """Client for Fireworks AI remote API."""

    def __init__(self, settings: Settings) -> None:
        allowed = [m.strip() for m in settings.allowed_models.split(",") if m.strip()]
        if not allowed:
            raise ValueError("ALLOWED_MODELS is empty — cannot select a Fireworks model")
        self.model = allowed[0]  # simplest choice for now: first listed model
        self.timeout = settings.remote_timeout_s
        self._client = AsyncOpenAI(
            base_url=settings.fireworks_base_url,
            api_key=settings.fireworks_api_key,
        )

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        last_exc = None
        for attempt in range(3):
            try:
                start = time.monotonic()
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **kwargs,
                )
                latency_ms = (time.monotonic() - start) * 1000

                if not response.choices:
                    raise RuntimeError(
                        f"Remote model returned no choices. Full response: {response}"
                    )

                choice = response.choices[0]
                usage = response.usage
                return LLMResponse(
                    text=choice.message.content or "",
                    tokens_used=usage.total_tokens if usage else 0,
                    latency_ms=round(latency_ms, 2),
                    model_name=self.model,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    async def health_check(self) -> bool:
        try:
            await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False