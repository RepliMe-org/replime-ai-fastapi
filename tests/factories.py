"""Builders for the request/DTO objects the tests exercise.

Keeping construction here means a schema change touches one place instead of
every test file.
"""
from schemas.chat import ChatbotConfig, ChatProcessRequest


class AsyncCM:
    """Minimal async context manager for faking aio_pika ``message.process()``."""

    def __init__(self, result=None):
        self.result = result

    async def __aenter__(self):
        return self.result

    async def __aexit__(self, *exc):
        return False


def make_config(**overrides) -> ChatbotConfig:
    data = dict(
        chatbot_name="TestBot",
        talk_like_me=False,
        tone="FRIENDLY",
        verbosity="BALANCED",
        formality="NEUTRAL",
        description=None,
        topics=None,
    )
    data.update(overrides)
    return ChatbotConfig(**data)


def make_chat_request(**overrides) -> ChatProcessRequest:
    data = dict(
        chatbot_id="cb-1",
        message_id=1,
        query="What is machine learning?",
        conversation_history=[],
        message_classes=[],
        config=make_config(),
        first_message=False,
    )
    if "config" in overrides and isinstance(overrides["config"], dict):
        overrides["config"] = make_config(**overrides["config"])
    data.update(overrides)
    return ChatProcessRequest(**data)


def make_chunk(**overrides) -> dict:
    """A retrieved-chunk dict shaped like VectorStore.search output."""
    data = dict(
        chunk_text="Machine learning is a subfield of AI.",
        youtube_video_id="vid1",
        video_title="Intro to ML",
        timestamp_seconds=10,
        similarity_score=0.9,
    )
    data.update(overrides)
    return data
