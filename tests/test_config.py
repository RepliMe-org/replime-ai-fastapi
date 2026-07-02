"""Unit tests for core/config.py — model routing and provider resolution."""
import pytest

from core.config import _LLM_PROVIDER_BASE_URLS, Settings, settings


def test_model_for_uses_arabic_override_when_corpus_arabic():
    assert Settings.model_for("base/model", "ar/model", "ar") == "ar/model"


def test_model_for_falls_back_to_base_when_no_override():
    assert Settings.model_for("base/model", "", "ar") == "base/model"


def test_model_for_uses_base_for_english_corpus():
    assert Settings.model_for("base/model", "ar/model", "en") == "base/model"


@pytest.mark.parametrize("provider", ["groq", "cerebras", "gemini", "nvidia"])
def test_provider_credentials_known_providers(provider):
    base_url, _key = settings.provider_credentials(provider)
    assert base_url == _LLM_PROVIDER_BASE_URLS[provider]


def test_provider_credentials_unknown_raises():
    with pytest.raises(ValueError):
        settings.provider_credentials("openai")
