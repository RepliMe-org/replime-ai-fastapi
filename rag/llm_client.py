import asyncio
import logging
import time

import groq as groq_module
from groq import Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from core.exceptions import LLMError

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    groq_module.APIConnectionError,
    groq_module.RateLimitError,
    groq_module.InternalServerError,
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _call_groq(
    client: Groq,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    start = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except _RETRYABLE_EXCEPTIONS:
        raise
    except groq_module.GroqError as exc:
        raise LLMError(str(exc)) from exc
    duration_ms = int((time.monotonic() - start) * 1000)
    return response.choices[0].message.content or "", duration_ms


class LLMClient:
    def __init__(self, api_key: str = settings.GROQ_API_KEY, model: str = settings.GROQ_CHAT_MODEL) -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, int]:
        try:
            text, duration_ms = await asyncio.to_thread(
                _call_groq, self._client, self._model, messages, max_tokens, temperature
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            raise LLMError(str(exc)) from exc
        logger.info("LLM generate: model=%s tokens=%s duration_ms=%d", self._model, max_tokens, duration_ms)
        return text, duration_ms


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
