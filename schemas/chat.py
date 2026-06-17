from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatbotConfig(BaseModel):
    chatbot_name: str
    talk_like_me: bool
    tone: str | None = None
    verbosity: str | None = None
    formality: str | None = None
    # Optional influencer-provided seed used to bootstrap the channel profile
    # for domain-aware intent before any video has been ingested.
    description: str | None = None
    topics: list[str] | None = None


class ConversationMessage(BaseModel):
    role: Literal["USER", "BOT"]
    content: str


class MessageClass(BaseModel):
    id: int
    name: str


class ChatProcessRequest(BaseModel):
    chatbot_id: str
    message_id: int | None = None
    query: str = Field(min_length=1, max_length=5000)
    conversation_history: list[ConversationMessage]
    message_classes: list[MessageClass]
    config: ChatbotConfig
    first_message: bool = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class Source(BaseModel):
    video_id: str
    video_title: str
    youtube_url: str


class ChatProcessResponse(BaseModel):
    answer: str
    session_title: str | None = None
    sources: list[Source]