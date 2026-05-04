from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatbotConfig(BaseModel):
    chatbot_name: str
    persona_description: str
    talk_like_me: bool
    tone: str | None = None
    verbosity: str
    formality: str | None = None
    default_language: str = "en"


class ConversationMessage(BaseModel):
    role: Literal["USER", "BOT"]
    content: str


class ChatProcessRequest(BaseModel):
    chatbot_id: str
    query: str = Field(min_length=1, max_length=5000)
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
    timestamp_seconds: int


class ChatProcessResponse(BaseModel):
    answer: str
    sources: list[Source]