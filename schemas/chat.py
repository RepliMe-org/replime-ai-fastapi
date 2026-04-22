from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatbotConfig(BaseModel):
    chatbot_name: str
    persona_description: str
    persona_keywords: list[str]
    tone: str
    response_length: str
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_context_turns: int = 10


class ConversationMessage(BaseModel):
    role: Literal["USER", "BOT"]
    content: str
    sent_at: datetime


class ChatProcessRequest(BaseModel):
    session_id: str
    chatbot_id: str
    query: str = Field(min_length=1, max_length=5000)
    language: str | None = None
    conversation_history: list[ConversationMessage]
    config: ChatbotConfig

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class Source(BaseModel):
    video_id: str
    video_title: str
    chunk_text: str
    youtube_url: str
    timestamp_seconds: int | None
    similarity_score: float


class ChatProcessResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: int
    llm_ms: int
    rewritten_query: str | None = None
