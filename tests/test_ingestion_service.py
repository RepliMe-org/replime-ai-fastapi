"""Service tests for services/ingestion_service.py — stage errors + callbacks."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.ingestion_service as ing
from core.exceptions import (
    NonRetryableIngestionError,
    RetryableIngestionError,
    TranscriptError,
    TranscriptRateLimitError,
    VectorStoreError,
)
from services.ingestion_service import run_ingestion, run_ingestion_pipeline


@pytest.fixture
def pipeline_deps(monkeypatch):
    monkeypatch.setattr(ing, "load_transcript", lambda vid: [{"text": "hello world content", "start": 0}])
    monkeypatch.setattr(ing, "detect_language", lambda text: "en")
    monkeypatch.setattr(
        ing, "chunk_transcript",
        lambda segments, language="en": [{"text": "chunk", "timestamp_seconds": 0}],
    )
    embedder = MagicMock()
    embedder.embed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    monkeypatch.setattr(ing, "get_embedder", lambda: embedder)
    vs = MagicMock()
    vs.upsert_chunks = MagicMock()
    monkeypatch.setattr(ing, "get_vector_store", lambda: vs)
    monkeypatch.setattr(ing, "refresh_corpus_language", AsyncMock())
    monkeypatch.setattr(ing, "spawn_channel_description_refresh", MagicMock())
    return SimpleNamespace(vs=vs, embedder=embedder)


async def test_pipeline_happy_path_upserts(pipeline_deps):
    await run_ingestion_pipeline("cb", "vid", "Title")
    pipeline_deps.vs.upsert_chunks.assert_called_once()


async def test_transcript_ratelimit_is_retryable(pipeline_deps, monkeypatch):
    monkeypatch.setattr(ing, "load_transcript", MagicMock(side_effect=TranscriptRateLimitError("blocked")))
    with pytest.raises(RetryableIngestionError) as ei:
        await run_ingestion_pipeline("cb", "vid", "t")
    assert ei.value.stage == "TRANSCRIPT_EXTRACTION"


async def test_transcript_error_is_nonretryable(pipeline_deps, monkeypatch):
    monkeypatch.setattr(ing, "load_transcript", MagicMock(side_effect=TranscriptError("no transcript")))
    with pytest.raises(NonRetryableIngestionError):
        await run_ingestion_pipeline("cb", "vid", "t")


async def test_unexpected_transcript_error_is_retryable(pipeline_deps, monkeypatch):
    monkeypatch.setattr(ing, "load_transcript", MagicMock(side_effect=RuntimeError("weird")))
    with pytest.raises(RetryableIngestionError):
        await run_ingestion_pipeline("cb", "vid", "t")


async def test_indexing_vectorstore_error_is_retryable(pipeline_deps):
    pipeline_deps.vs.upsert_chunks.side_effect = VectorStoreError("qdrant down")
    with pytest.raises(RetryableIngestionError) as ei:
        await run_ingestion_pipeline("cb", "vid", "t")
    assert ei.value.stage == "VECTOR_INDEXING"


async def test_run_ingestion_success_reports_completed(monkeypatch):
    monkeypatch.setattr(ing, "run_ingestion_pipeline", AsyncMock())
    callback = AsyncMock()
    monkeypatch.setattr(ing, "send_ingestion_callback", callback)
    await run_ingestion("cb", "vid", "t")
    assert callback.await_args.args[1]["status"] == "COMPLETED"


async def test_run_ingestion_nonretryable_reports_failed_permanent(monkeypatch):
    monkeypatch.setattr(
        ing, "run_ingestion_pipeline",
        AsyncMock(side_effect=NonRetryableIngestionError("TRANSCRIPT_EXTRACTION", "no transcript")),
    )
    callback = AsyncMock()
    monkeypatch.setattr(ing, "send_ingestion_callback", callback)
    await run_ingestion("cb", "vid", "t")
    payload = callback.await_args.args[1]
    assert payload["status"] == "FAILED"
    assert payload["retryable"] is False


async def test_run_ingestion_retryable_reports_failed_retryable(monkeypatch):
    monkeypatch.setattr(
        ing, "run_ingestion_pipeline",
        AsyncMock(side_effect=RetryableIngestionError("EMBEDDING_GENERATION", "timeout")),
    )
    callback = AsyncMock()
    monkeypatch.setattr(ing, "send_ingestion_callback", callback)
    await run_ingestion("cb", "vid", "t")
    payload = callback.await_args.args[1]
    assert payload["status"] == "FAILED"
    assert payload["retryable"] is True


async def test_send_ingestion_callback_swallows_errors(monkeypatch):
    monkeypatch.setattr(ing, "_do_callback", AsyncMock(side_effect=RuntimeError("boom")))
    # Must not raise — the video is already indexed regardless of callback outcome.
    await ing.send_ingestion_callback("vid", {"status": "COMPLETED"})
