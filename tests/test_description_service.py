"""Service tests for services/description_service.py — regeneration + locking."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.description_service as desc


@pytest.fixture
def regen_deps(monkeypatch):
    vs = MagicMock()
    vs.sample_chunks = MagicMock(
        return_value=[{"video_title": "T", "excerpts": ["some excerpt text about testing"]}]
    )
    monkeypatch.setattr(desc, "get_vector_store", lambda: vs)
    monkeypatch.setattr(desc, "get_corpus_language", AsyncMock(return_value="en"))
    gen = MagicMock()
    gen.regenerate = AsyncMock(return_value="A channel about software testing.")
    monkeypatch.setattr(desc, "get_description_generator", lambda spec: gen)
    send = AsyncMock()
    monkeypatch.setattr(desc, "send_description_callback", send)
    return SimpleNamespace(vs=vs, gen=gen, send=send)


async def test_regenerate_reports_description(regen_deps):
    result = await desc._regenerate_and_report("cb")
    assert result == "A channel about software testing."
    regen_deps.send.assert_awaited_once_with("cb", "A channel about software testing.")


async def test_regenerate_empty_sample_clears_description(regen_deps):
    regen_deps.vs.sample_chunks.return_value = []
    result = await desc._regenerate_and_report("cb")
    assert result is None
    regen_deps.send.assert_awaited_once_with("cb", None)


async def test_refresh_falls_back_to_local_lock_when_redis_down(monkeypatch):
    monkeypatch.setattr(desc, "get_redis", AsyncMock(side_effect=RuntimeError("no redis")))
    regen = AsyncMock(return_value="generated description")
    monkeypatch.setattr(desc, "_regenerate_and_report", regen)
    result = await desc.refresh_channel_description("cb")
    assert result == "generated description"
    regen.assert_awaited_once_with("cb")


async def test_refresh_swallows_regeneration_errors(monkeypatch):
    monkeypatch.setattr(desc, "get_redis", AsyncMock(side_effect=RuntimeError("no redis")))
    monkeypatch.setattr(desc, "_regenerate_and_report", AsyncMock(side_effect=RuntimeError("boom")))
    # Best-effort: a regeneration failure must not propagate.
    assert await desc.refresh_channel_description("cb") is None
