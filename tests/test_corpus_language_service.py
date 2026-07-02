"""Service tests for services/corpus_language_service.py — cache read/refresh."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.corpus_language_service as clang


@pytest.fixture
def redis_and_vs(monkeypatch):
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    monkeypatch.setattr(clang, "get_redis", AsyncMock(return_value=redis))
    vs = MagicMock()
    vs.dominant_language = MagicMock(return_value="en")
    monkeypatch.setattr(clang, "get_vector_store", lambda: vs)
    return SimpleNamespace(redis=redis, vs=vs)


async def test_returns_cached_value_without_recompute(redis_and_vs):
    redis_and_vs.redis.get.return_value = "ar"
    assert await clang.get_corpus_language("cb") == "ar"
    redis_and_vs.vs.dominant_language.assert_not_called()


async def test_recomputes_and_caches_on_miss(redis_and_vs):
    redis_and_vs.redis.get.return_value = None
    redis_and_vs.vs.dominant_language.return_value = "ar"
    assert await clang.get_corpus_language("cb") == "ar"
    redis_and_vs.redis.set.assert_awaited()


async def test_defaults_to_en_when_recompute_fails(redis_and_vs):
    redis_and_vs.redis.get.return_value = None
    redis_and_vs.vs.dominant_language.side_effect = RuntimeError("qdrant down")
    assert await clang.get_corpus_language("cb") == "en"


async def test_refresh_caches_dominant_language(redis_and_vs):
    redis_and_vs.vs.dominant_language.return_value = "ar"
    assert await clang.refresh_corpus_language("cb") == "ar"
    redis_and_vs.redis.set.assert_awaited()


async def test_refresh_returns_none_on_failure(redis_and_vs):
    redis_and_vs.vs.dominant_language.side_effect = RuntimeError("boom")
    assert await clang.refresh_corpus_language("cb") is None
