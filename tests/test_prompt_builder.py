"""Unit tests for rag/llm/prompt_builder.py — persona + message assembly."""
from rag.llm.prompt_builder import _format_chunks, build_messages, build_system_prompt
from schemas.chat import ConversationMessage
from tests.factories import make_config


def test_talk_like_me_persona():
    sp = build_system_prompt(make_config(talk_like_me=True), "en")
    assert "TestBot" in sp
    assert "mirror" in sp.lower()


def test_configured_persona_maps_tone_formality_verbosity():
    sp = build_system_prompt(
        make_config(tone="FRIENDLY", formality="FORMAL", verbosity="CONCISE"), "en"
    )
    assert "warm and approachable" in sp
    assert "formal, professional" in sp
    assert "brief and to the point" in sp


def test_verbosity_none_does_not_crash():
    # Regression: verbosity is Optional in the schema; None must not raise.
    sp = build_system_prompt(make_config(verbosity=None), "en")
    assert "TestBot" in sp


def test_answer_rules_inject_reply_language():
    assert "Arabic" in build_system_prompt(make_config(), "ar")


def test_format_chunks_with_timestamp():
    out = _format_chunks(
        [{"chunk_text": "t", "youtube_video_id": "v", "video_title": "T", "timestamp_seconds": 65}]
    )
    assert "[1]" in out
    assert "01:05" in out
    assert "&t=65s" in out


def test_format_chunks_without_timestamp():
    out = _format_chunks(
        [{"chunk_text": "t", "youtube_video_id": "v", "video_title": "T", "timestamp_seconds": None}]
    )
    assert "00:00" in out
    assert "&t=" not in out


def test_build_messages_structure():
    chunks = [
        {"chunk_text": "txt", "youtube_video_id": "v", "video_title": "T", "timestamp_seconds": 65}
    ]
    history = [
        ConversationMessage(role="USER", content="hi"),
        ConversationMessage(role="BOT", content="hello"),
    ]
    msgs = build_messages("My question", chunks, history, make_config(), "en")
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2] == {"role": "assistant", "content": "hello"}
    assert msgs[-1]["role"] == "user"
    assert "<context>" in msgs[-1]["content"]
    assert "<question>\nMy question\n</question>" in msgs[-1]["content"]


def test_build_messages_without_chunks_uses_plain_question():
    msgs = build_messages("Just this", [], [], make_config(), "en")
    assert msgs[-1]["content"] == "Just this"
