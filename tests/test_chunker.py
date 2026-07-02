"""Unit tests for rag/text/chunker.py — transcript chunking + timestamp mapping."""
import pytest

from core.exceptions import TranscriptError
from rag.text.chunker import chunk_transcript


def test_empty_segments_returns_empty():
    assert chunk_transcript([]) == []


def test_english_chunking_keeps_first_timestamp():
    segments = [
        {"text": "Hello world.", "start": 0},
        {"text": "This is a test transcript.", "start": 5},
    ]
    out = chunk_transcript(segments)
    assert len(out) >= 1
    assert out[0]["timestamp_seconds"] == 0
    assert "Hello world." in out[0]["text"]


def test_whitespace_only_transcript_raises():
    with pytest.raises(TranscriptError):
        chunk_transcript([{"text": "   ", "start": 0}])


def test_missing_start_yields_none_timestamp():
    out = chunk_transcript([{"text": "Some sufficiently long transcript text for one chunk."}])
    assert out[0]["timestamp_seconds"] is None


def test_timestamps_are_integers():
    out = chunk_transcript([{"text": "word " * 60, "start": 1.9}])
    assert all(isinstance(c["timestamp_seconds"], int) for c in out)
    assert out[0]["timestamp_seconds"] == 1  # int(1.9)


def test_arabic_normalization_applied_to_chunks():
    segments = [
        {"text": "هذا نص عربي فيه كلمة الأمثلة ويجب أن يكون طويلا بما يكفي للاختبار", "start": 0}
    ]
    out = chunk_transcript(segments, language="ar")
    joined = " ".join(c["text"] for c in out)
    assert "أ" not in joined  # alef-hamza normalized to bare alef
    assert "ا" in joined
