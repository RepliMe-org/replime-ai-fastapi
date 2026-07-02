"""Service tests for services/analytics_service.py — clustering + JSON parsing."""
from unittest.mock import AsyncMock, MagicMock

import services.analytics_service as an
from schemas.analytics import AnalyticsRequest, QuestionInput
from services.analytics_service import _parse_llm_json, compute_analytics


def test_parse_plain_json():
    assert _parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    assert _parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_from_surrounding_text():
    assert _parse_llm_json('sure, here: {"a": 1} done') == {"a": 1}


def test_parse_invalid_returns_empty_dict():
    assert _parse_llm_json("not json at all") == {}


async def test_empty_questions_returns_empty_response():
    resp = await compute_analytics(AnalyticsRequest(chatbot_id="cb"))
    assert resp.most_asked_clusters == []
    assert resp.content_gaps == []
    assert resp.executive_summary == ""


async def test_clusters_and_gaps_parsed(monkeypatch):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=(
        '{"mostAskedClusters":[{"theme":"Pricing","count":3,"exampleQuestions":["How much?"]}],'
        '"contentGaps":[{"topic":"Refunds","frequency":2,"sampleQuestions":["Refund?"]}],'
        '"executiveSummary":"Users ask about pricing."}', 10
    ))
    monkeypatch.setattr(an, "get_client_for", lambda spec: llm)
    monkeypatch.setattr(an, "get_corpus_language", AsyncMock(return_value="en"))
    req = AnalyticsRequest(chatbot_id="cb", questions=[QuestionInput(text="How much?")])
    resp = await compute_analytics(req)
    assert resp.most_asked_clusters[0].theme == "Pricing"
    assert resp.most_asked_clusters[0].count == 3
    assert resp.content_gaps[0].topic == "Refunds"
    assert resp.executive_summary == "Users ask about pricing."


async def test_llm_failure_returns_empty_response(monkeypatch):
    llm = MagicMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    monkeypatch.setattr(an, "get_client_for", lambda spec: llm)
    monkeypatch.setattr(an, "get_corpus_language", AsyncMock(return_value="en"))
    req = AnalyticsRequest(chatbot_id="cb", questions=[QuestionInput(text="Q")])
    resp = await compute_analytics(req)
    assert resp.most_asked_clusters == []
    assert resp.executive_summary == ""
