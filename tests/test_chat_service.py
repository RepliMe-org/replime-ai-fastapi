"""Service tests for services/chat_service.py — the full RAG chat pipeline.

Every external boundary (corpus language, rewriter, title/intent, embedder,
vector store, LLM) is mocked, so these assert pipeline orchestration only.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.chat_service as cs
from tests.factories import make_chat_request, make_chunk, make_config


@pytest.fixture
def deps(monkeypatch):
    corpus = AsyncMock(return_value="en")
    monkeypatch.setattr(cs, "get_corpus_language", corpus)

    rewriter = MagicMock()
    rewriter.rewrite = AsyncMock(side_effect=lambda q, h, language="en": q)
    monkeypatch.setattr(cs, "get_query_rewriter", lambda spec: rewriter)

    title = MagicMock()
    title.generate = AsyncMock(return_value="Generated Title")
    monkeypatch.setattr(cs, "get_title_generator", lambda: title)

    intent = MagicMock()
    intent.classify = AsyncMock(return_value="CONTENT_QUESTION")
    monkeypatch.setattr(cs, "get_intent_classifier", lambda: intent)

    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    monkeypatch.setattr(cs, "get_embedder", lambda: embedder)

    vs = MagicMock()
    vs.search = MagicMock(return_value=[])
    monkeypatch.setattr(cs, "get_vector_store", lambda: vs)

    llm = MagicMock()
    llm.generate = AsyncMock(return_value=("Answer text [1]", 12))
    monkeypatch.setattr(cs, "get_client_for", lambda spec: llm)

    return SimpleNamespace(
        corpus=corpus, rewriter=rewriter, title=title, intent=intent,
        embedder=embedder, vs=vs, llm=llm,
    )


async def test_greeting_short_circuits_before_retrieval(deps):
    deps.intent.classify.return_value = "GREETING"
    resp = await cs.process_chat(make_chat_request(query="hello there"))
    assert resp.intent == "GREETING"
    assert resp.sources == []
    assert "TestBot" in resp.answer
    deps.embedder.embed_query.assert_not_called()
    deps.vs.search.assert_not_called()


async def test_content_question_without_chunks_returns_fallback(deps):
    deps.intent.classify.return_value = "CONTENT_QUESTION"
    deps.vs.search.return_value = []
    resp = await cs.process_chat(make_chat_request())
    assert resp.sources == []
    assert "TestBot" in resp.answer
    deps.llm.generate.assert_not_called()


async def test_content_question_happy_path_builds_sources(deps):
    deps.intent.classify.return_value = "CONTENT_QUESTION"
    deps.vs.search.return_value = [make_chunk(youtube_video_id="vid1", timestamp_seconds=30)]
    deps.llm.generate.return_value = ("Here is the answer [1].", 20)
    resp = await cs.process_chat(make_chat_request())
    assert resp.intent == "CONTENT_QUESTION"
    assert len(resp.sources) == 1
    assert resp.sources[0].video_id == "vid1"
    assert "t=30s" in resp.sources[0].youtube_url
    assert "[1]" not in resp.answer  # citation markers stripped


async def test_sources_deduped_keeping_highest_score(deps):
    deps.intent.classify.return_value = "CONTENT_QUESTION"
    deps.vs.search.return_value = [
        make_chunk(youtube_video_id="vidA", similarity_score=0.9, timestamp_seconds=10),
        make_chunk(youtube_video_id="vidA", similarity_score=0.5, timestamp_seconds=99),
    ]
    deps.llm.generate.return_value = ("Answer [1][2].", 20)
    resp = await cs.process_chat(make_chat_request())
    assert len(resp.sources) == 1
    assert "t=10s" in resp.sources[0].youtube_url  # highest-score chunk wins


async def test_first_message_generates_title(deps):
    deps.intent.classify.return_value = "CONTENT_QUESTION"
    deps.vs.search.return_value = []
    resp = await cs.process_chat(make_chat_request(first_message=True))
    assert resp.session_title == "Generated Title"


async def test_first_message_non_content_uses_mapped_title(deps):
    deps.intent.classify.return_value = "GREETING"
    resp = await cs.process_chat(make_chat_request(first_message=True, query="hello there"))
    assert resp.session_title == "Greeting"


async def test_harmful_intent_has_no_title_or_sources(deps):
    deps.intent.classify.return_value = "HARMFUL"
    resp = await cs.process_chat(make_chat_request(first_message=True, query="please do bad things"))
    assert resp.intent == "HARMFUL"
    assert resp.session_title is None
    assert resp.sources == []


def test_seed_description_combines_description_and_topics():
    out = cs._seed_description(make_config(description="A channel about X", topics=["a", "b"]))
    assert "A channel about X" in out
    assert "Topics: a, b" in out


def test_seed_description_none_when_empty():
    assert cs._seed_description(make_config(description=None, topics=None)) is None
