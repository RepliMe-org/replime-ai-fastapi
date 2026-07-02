"""Retrieval tests for rag/retrieval/vector_store.py with a faked Qdrant client.

The real QdrantClient and the sparse BM25 model are replaced with mocks so these
tests assert query construction, result mapping, dedup counting, and the
dominant-language ratio without a live Qdrant instance.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.config import settings
from core.exceptions import VectorStoreError
from rag.retrieval.vector_store import VectorStore, _denoise_chunks, _evenly_spaced


def _vs_with_fakes() -> VectorStore:
    vs = VectorStore("http://fake", "", "test")
    vs._client = MagicMock()
    vs._sparse_model = MagicMock()
    vs._sparse_model.query_embed.return_value = iter(
        [SimpleNamespace(indices=np.array([1, 2]), values=np.array([0.5, 0.5]))]
    )
    return vs


def _point(vid="v1", title="T", ts=10, score=0.9, text="chunk text"):
    return SimpleNamespace(
        payload={
            "chunk_text": text,
            "youtube_video_id": vid,
            "video_title": title,
            "timestamp_seconds": ts,
        },
        score=score,
        vector=None,
    )


# ── pure helpers ────────────────────────────────────────────────────────────

def test_evenly_spaced_samples_across_list():
    assert _evenly_spaced(["a", "b", "c", "d"], 2) == ["a", "c"]
    assert _evenly_spaced(["a", "b"], 5) == ["a", "b"]
    assert _evenly_spaced([], 3) == []


def test_denoise_drops_short_dupes_and_outro():
    texts = ["short"] + [c * 100 for c in "abcde"] + ["a" * 100]  # 5 distinct + 1 dup + 1 short
    out = _denoise_chunks(texts)
    assert all(len(t) >= 80 for t in out)
    assert len(out) == len(set(out))
    assert len(out) == 4  # dup + short removed (6→5), then final outro chunk dropped


# ── search ──────────────────────────────────────────────────────────────────

def test_search_maps_points(monkeypatch):
    monkeypatch.setattr(settings, "USE_MMR", False)
    vs = _vs_with_fakes()
    vs._client.query_points.return_value = SimpleNamespace(points=[_point(vid="a"), _point(vid="b")])
    out = vs.search("cb", "query", [0.1, 0.2], top_k=5, similarity_threshold=0.4)
    assert [c["youtube_video_id"] for c in out] == ["a", "b"]
    assert out[0]["similarity_score"] == 0.9


def test_search_wraps_errors(monkeypatch):
    monkeypatch.setattr(settings, "USE_MMR", False)
    vs = _vs_with_fakes()
    vs._client.query_points.side_effect = RuntimeError("qdrant boom")
    with pytest.raises(VectorStoreError):
        vs.search("cb", "q", [0.1], 5, 0.4)


# ── delete ──────────────────────────────────────────────────────────────────

def test_delete_counts_then_deletes():
    vs = _vs_with_fakes()
    vs._client.count.return_value = SimpleNamespace(count=3)
    assert vs.delete_by_video_id("cb", "vid") == 3
    vs._client.delete.assert_called_once()


def test_delete_is_noop_when_nothing_matches():
    vs = _vs_with_fakes()
    vs._client.count.return_value = SimpleNamespace(count=0)
    assert vs.delete_by_video_id("cb", "vid") == 0
    vs._client.delete.assert_not_called()


# ── dominant language ───────────────────────────────────────────────────────

def test_dominant_language_empty_corpus_is_english():
    vs = _vs_with_fakes()
    vs._client.count.return_value = SimpleNamespace(count=0)
    assert vs.dominant_language("cb") == "en"


def test_dominant_language_arabic_over_threshold(monkeypatch):
    monkeypatch.setattr(settings, "CORPUS_ARABIC_RATIO_THRESHOLD", 0.1)
    vs = _vs_with_fakes()
    vs._client.count.side_effect = [SimpleNamespace(count=100), SimpleNamespace(count=50)]
    assert vs.dominant_language("cb") == "ar"


def test_dominant_language_below_threshold_is_english(monkeypatch):
    monkeypatch.setattr(settings, "CORPUS_ARABIC_RATIO_THRESHOLD", 0.5)
    vs = _vs_with_fakes()
    vs._client.count.side_effect = [SimpleNamespace(count=100), SimpleNamespace(count=10)]
    assert vs.dominant_language("cb") == "en"


# ── sampling / listing / health ─────────────────────────────────────────────

def test_sample_chunks_returns_per_video_structure():
    vs = _vs_with_fakes()
    points = [
        SimpleNamespace(payload={"youtube_video_id": "v1", "video_title": "T1", "chunk_text": "x" * 100}),
        SimpleNamespace(payload={"youtube_video_id": "v1", "video_title": "T1", "chunk_text": "y" * 100}),
    ]
    vs._client.scroll.return_value = (points, None)
    out = vs.sample_chunks("cb", per_video=6, max_chars=10000)
    assert out[0]["video_title"] == "T1"
    assert "excerpts" in out[0]


def test_list_videos_groups_by_chatbot():
    vs = _vs_with_fakes()
    points = [
        SimpleNamespace(payload={"chatbot_id": "cb1", "youtube_video_id": "v1", "video_title": "A"}),
        SimpleNamespace(payload={"chatbot_id": "cb1", "youtube_video_id": "v1", "video_title": "A"}),
        SimpleNamespace(payload={"chatbot_id": "cb2", "youtube_video_id": "v2", "video_title": "B"}),
    ]
    vs._client.scroll.return_value = (points, None)
    grouped = vs.list_videos()
    assert set(grouped.keys()) == {"cb1", "cb2"}
    assert grouped["cb1"][0]["chunk_count"] == 2


def test_healthcheck_pings_qdrant():
    vs = _vs_with_fakes()
    vs.healthcheck()
    vs._client.get_collections.assert_called_once()
