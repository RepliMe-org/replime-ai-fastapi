"""Contract tests for the Pydantic schemas at the Spring Boot boundary."""
import pytest
from pydantic import ValidationError

from schemas.analytics import AnalyticsRequest, AnalyticsResponse, QuestionInput
from schemas.chat import ChatProcessRequest
from tests.factories import make_chat_request


def test_valid_chat_request():
    req = make_chat_request()
    assert req.chatbot_id == "cb-1"


def test_blank_query_rejected():
    with pytest.raises(ValidationError):
        make_chat_request(query="   ")


def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        make_chat_request(query="")


def test_overlong_query_rejected():
    with pytest.raises(ValidationError):
        make_chat_request(query="x" * 5001)


def test_invalid_conversation_role_rejected():
    with pytest.raises(ValidationError):
        ChatProcessRequest(
            chatbot_id="cb",
            query="hi",
            conversation_history=[{"role": "ADMIN", "content": "x"}],
            message_classes=[],
            config={"chatbot_name": "B", "talk_like_me": False},
        )


def test_analytics_question_accepts_camelcase_alias():
    q = QuestionInput.model_validate({"text": "How?", "answeredWithSources": False})
    assert q.answered_with_sources is False


def test_analytics_question_defaults_answered_true():
    assert QuestionInput.model_validate({"text": "How?"}).answered_with_sources is True


def test_analytics_request_parses_camelcase():
    req = AnalyticsRequest.model_validate(
        {"chatbotId": "cb-9", "questions": [{"text": "Q1"}]}
    )
    assert req.chatbot_id == "cb-9"
    assert len(req.questions) == 1


def test_analytics_response_serializes_camelcase():
    dumped = AnalyticsResponse().model_dump(by_alias=True)
    assert "mostAskedClusters" in dumped
    assert "contentGaps" in dumped
    assert "executiveSummary" in dumped
