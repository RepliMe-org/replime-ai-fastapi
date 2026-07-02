"""API tests for POST /ai/chat/process."""
from unittest.mock import AsyncMock

import routes.chat as chat_route
from schemas.chat import ChatProcessResponse


def _canned_response(intent="GREETING"):
    return ChatProcessResponse(
        answer="Hi there", session_title="A title", sources=[], intent=intent, message_id=1
    )


def _payload(**overrides):
    data = dict(
        chatbot_id="cb-1",
        message_id=1,
        query="hello",
        conversation_history=[],
        message_classes=[],
        config={"chatbot_name": "Bot", "talk_like_me": False},
        first_message=True,
    )
    data.update(overrides)
    return data


def test_chat_process_returns_response(app_client, monkeypatch):
    monkeypatch.setattr(chat_route, "process_chat", AsyncMock(return_value=_canned_response("GREETING")))
    monkeypatch.setattr(chat_route, "classify_and_report", AsyncMock())
    r = app_client.post("/ai/chat/process", json=_payload())
    assert r.status_code == 200
    assert r.json()["intent"] == "GREETING"


def test_chat_process_schedules_classification_when_classes_present(app_client, monkeypatch):
    classify = AsyncMock()
    monkeypatch.setattr(chat_route, "process_chat", AsyncMock(return_value=_canned_response()))
    monkeypatch.setattr(chat_route, "classify_and_report", classify)
    r = app_client.post("/ai/chat/process", json=_payload(message_classes=[{"id": 1, "name": "Billing"}]))
    assert r.status_code == 200
    classify.assert_awaited_once()


def test_chat_process_skips_classification_without_classes(app_client, monkeypatch):
    classify = AsyncMock()
    monkeypatch.setattr(chat_route, "process_chat", AsyncMock(return_value=_canned_response()))
    monkeypatch.setattr(chat_route, "classify_and_report", classify)
    app_client.post("/ai/chat/process", json=_payload(message_classes=[]))
    classify.assert_not_called()


def test_chat_process_rejects_blank_query(app_client):
    r = app_client.post("/ai/chat/process", json=_payload(query="   "))
    assert r.status_code == 422


def test_chat_process_rejects_missing_query(app_client):
    payload = _payload()
    del payload["query"]
    assert app_client.post("/ai/chat/process", json=payload).status_code == 422
