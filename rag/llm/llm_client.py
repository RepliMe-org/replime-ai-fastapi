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

# A model is abandoned for the next one in the chain on any of these — the
# retryable errors (after per-model retries are exhausted) plus any LLMError
# raised for a non-retryable provider error.
_FALLBACK_TRIGGERS = _RETRYABLE_EXCEPTIONS + (LLMError,)


def _parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Split a "provider/model" spec into (provider, model)."""
    provider, _, model = model_spec.partition("/")
    if not provider or not model:
        raise LLMError(f"Invalid model spec {model_spec!r}; expected 'provider/model'")
    return provider, model


@lru_cache(maxsize=None)
def _provider_client(provider: str) -> OpenAI:
    """One OpenAI client (and connection pool) per provider, shared across all
    tasks and fallback chains. Pass a placeholder when the key is unset so
    construction succeeds; the provider returns a clear auth error on first use."""
    base_url, api_key = settings.provider_credentials(provider)
    return OpenAI(base_url=base_url, api_key=api_key or "EMPTY")


def _resolve_chain(primary_spec: str) -> list[str]:
    """Primary spec followed by the configured global fallbacks, de-duplicated
    with order preserved. Empty LLM_FALLBACK_MODELS ⇒ just the primary (so the
    behaviour is identical to no fallback)."""
    ordered: list[str] = [primary_spec]
    for spec in settings.fallback_model_specs():
        if spec and spec not in ordered:
            ordered.append(spec)
    return ordered


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
    """Calls a primary model, transparently falling back to the configured
    fallback models (in order) when it errors. The model that actually produced
    the response is logged; the caller only sees an error if every model fails."""

    def __init__(self, model_spec: str = settings.CHAT_MODEL) -> None:
        # Each entry is (provider, model, spec); parsing here surfaces an invalid
        # primary spec at construction, as before.
        self._chain = [
            (*_parse_model_spec(spec), spec) for spec in _resolve_chain(model_spec)
        ]

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, int]:
        last_error: Exception | None = None
        for index, (provider, model, spec) in enumerate(self._chain):
            try:
                text, duration_ms = await asyncio.to_thread(
                    _call_llm, _provider_client(provider), model, messages, max_tokens, temperature
                )
            except _FALLBACK_TRIGGERS as exc:
                last_error = exc
                next_spec = self._chain[index + 1][2] if index + 1 < len(self._chain) else None
                if next_spec:
                    logger.warning(
                        "LLM model %s failed (%s); falling back to %s", spec, exc, next_spec
                    )
                else:
                    logger.error("LLM model %s failed (%s); no fallback left", spec, exc)
                continue
            logger.info(
                "LLM generate: model=%s (%s) max_tokens=%s duration_ms=%d",
                spec,
                "primary" if index == 0 else f"fallback #{index}",
                max_tokens,
                duration_ms,
            )
            return text, duration_ms
        raise LLMError(
            f"All models failed ({', '.join(spec for _, _, spec in self._chain)})"
        ) from last_error


@lru_cache(maxsize=None)
def get_client_for(model_spec: str) -> LLMClient:
    """Process-wide cached client for a 'provider/model' spec; tasks on the
    same model share one client (and its fallback chain)."""
    return LLMClient(model_spec)
