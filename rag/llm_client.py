import asyncio
import logging
import time

from groq import Groq, GroqError

from core.config import settings
from core.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str = settings.GROQ_API_KEY, model: str = settings.GROQ_MODEL) -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, int]:
        def _call() -> tuple[str, int]:
            start = time.monotonic()
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except GroqError as exc:
                raise LLMError(str(exc)) from exc
            duration_ms = int((time.monotonic() - start) * 1000)
            text = response.choices[0].message.content or ""
            return text, duration_ms

        text, duration_ms = await asyncio.to_thread(_call)
        logger.info("LLM generate: model=%s tokens=%s duration_ms=%d", self._model, max_tokens, duration_ms)
        return text, duration_ms


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
