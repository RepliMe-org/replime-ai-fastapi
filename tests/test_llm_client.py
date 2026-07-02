"""Unit tests for rag/llm/llm_client.py — spec parsing, fallback chain, errors."""
import httpx
import openai
import pytest

import rag.llm.llm_client as llm_client_mod
from core.config import settings
from core.exceptions import LLMError
from rag.llm.llm_client import LLMClient, _parse_model_spec, _resolve_chain, get_client_for


def _conn_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://test"))


def test_parse_model_spec_simple():
    assert _parse_model_spec("groq/llama-3.1-8b-instant") == ("groq", "llama-3.1-8b-instant")


def test_parse_model_spec_nvidia_double_prefixed():
    # NVIDIA model ids contain their own "/", so only the first "/" splits.
    assert _parse_model_spec("nvidia/qwen/qwen3-235b") == ("nvidia", "qwen/qwen3-235b")


@pytest.mark.parametrize("bad", ["", "nomodel", "/model", "provider/"])
def test_parse_model_spec_invalid_raises(bad):
    with pytest.raises(LLMError):
        _parse_model_spec(bad)


def test_resolve_chain_without_fallback(monkeypatch):
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODELS", "")
    assert _resolve_chain("groq/primary") == ["groq/primary"]


def test_resolve_chain_dedupes_and_preserves_order(monkeypatch):
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODELS", "groq/primary, cerebras/backup")
    assert _resolve_chain("groq/primary") == ["groq/primary", "cerebras/backup"]


def test_client_builds_chain(monkeypatch):
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODELS", "")
    client = LLMClient("groq/llama-3.1-8b-instant")
    assert client._chain == [("groq", "llama-3.1-8b-instant", "groq/llama-3.1-8b-instant")]


async def test_generate_returns_text_and_duration(monkeypatch):
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODELS", "")
    monkeypatch.setattr(llm_client_mod, "_call_llm", lambda *a, **k: ("hello there", 42))
    text, ms = await LLMClient("groq/model").generate([{"role": "user", "content": "hi"}])
    assert text == "hello there"
    assert ms == 42


async def test_generate_falls_back_to_next_model(monkeypatch):
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODELS", "cerebras/backup")
    seen = []

    def fake_call(client, model, messages, max_tokens, temperature):
        seen.append(model)
        if model == "primary":
            raise _conn_error()
        return ("answer from backup", 7)

    monkeypatch.setattr(llm_client_mod, "_call_llm", fake_call)
    text, _ = await LLMClient("groq/primary").generate([{"role": "user", "content": "hi"}])
    assert text == "answer from backup"
    assert seen == ["primary", "backup"]


async def test_generate_raises_llmerror_when_all_models_fail(monkeypatch):
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODELS", "")

    def boom(*a, **k):
        raise _conn_error()

    monkeypatch.setattr(llm_client_mod, "_call_llm", boom)
    with pytest.raises(LLMError):
        await LLMClient("groq/model").generate([{"role": "user", "content": "hi"}])


def test_get_client_for_is_cached():
    assert get_client_for("groq/cache-me") is get_client_for("groq/cache-me")
