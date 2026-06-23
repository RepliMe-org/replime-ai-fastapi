import asyncio
import logging
import time
from functools import lru_cache

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from core.exceptions import LLMError

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Split a "provider/model" spec into (provider, model)."""
    provider, _, model = model_spec.partition("/")
    if not provider or not model:
        raise LLMError(f"Invalid model spec {model_spec!r}; expected 'provider/model'")
    return provider, model


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _call_llm(
    client: OpenAI,
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
    except openai.OpenAIError as exc:
        raise LLMError(str(exc)) from exc
    duration_ms = int((time.monotonic() - start) * 1000)
    return response.choices[0].message.content or "", duration_ms


class LLMClient:
    def __init__(self, model_spec: str = settings.CHAT_MODEL) -> None:
        self._provider, self._model = _parse_model_spec(model_spec)
        base_url, api_key = settings.provider_credentials(self._provider)
        # Pass a placeholder when the key is unset so construction succeeds;
        # the provider returns a clear auth error on the first real call.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, int]:
        try:
            text, duration_ms = await asyncio.to_thread(
                _call_llm, self._client, self._model, messages, max_tokens, temperature
            )
        except _RETRYABLE_EXCEPTIONS as exc:
            raise LLMError(str(exc)) from exc
        logger.info(
            "LLM generate: provider=%s model=%s tokens=%s duration_ms=%d",
            self._provider,
            self._model,
            max_tokens,
            duration_ms,
        )
        return text, duration_ms


@lru_cache(maxsize=None)
def get_client_for(model_spec: str) -> LLMClient:
    """Process-wide cached client for a 'provider/model' spec; tasks on the
    same model share one client (and its connection pool)."""
    return LLMClient(model_spec)


def get_llm_client() -> LLMClient:
    return get_client_for(settings.CHAT_MODEL)
