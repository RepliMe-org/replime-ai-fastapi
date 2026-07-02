"""API tests for POST /ai/analytics/process — camelCase response contract."""
from unittest.mock import AsyncMock

import routes.analytics as an_route
from schemas.analytics import AnalyticsResponse, QuestionCluster


def test_analytics_process_returns_camelcase_body(app_client, monkeypatch):
    resp = AnalyticsResponse(
        most_asked_clusters=[QuestionCluster(theme="Pricing", count=2, example_questions=["How much?"])],
        executive_summary="Audience asks about pricing.",
    )
    monkeypatch.setattr(an_route, "compute_analytics", AsyncMock(return_value=resp))
    r = app_client.post(
        "/ai/analytics/process",
        json={"chatbotId": "cb", "questions": [{"text": "How much?"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mostAskedClusters"][0]["theme"] == "Pricing"
    assert body["executiveSummary"] == "Audience asks about pricing."


def test_analytics_process_accepts_empty_questions(app_client, monkeypatch):
    monkeypatch.setattr(an_route, "compute_analytics", AsyncMock(return_value=AnalyticsResponse()))
    r = app_client.post("/ai/analytics/process", json={"chatbotId": "cb", "questions": []})
    assert r.status_code == 200
    assert r.json()["mostAskedClusters"] == []
