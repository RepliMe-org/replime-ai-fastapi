from typing import Literal
from pydantic import BaseModel


# ── Inbound (FastAPI receives from Spring Boot via HTTP) ───────────────────

class VideoInput(BaseModel):
    youtube_video_id: str
    video_title: str


class IndexVideosRequest(BaseModel):
    chatbot_id: str
    videos: list[VideoInput]


# ── Outbound: 202 response (FastAPI → caller, immediately) ────────────────

class IndexVideosAcceptedResponse(BaseModel):
    status: Literal["ACCEPTED"]
    chatbot_id: str
    total: int


class DeleteVideoRequest(BaseModel):
    chatbot_id: str
    youtube_video_id: str


class DeleteVideoResponse(BaseModel):
    youtube_video_id: str
    deleted_chunks: int


class VideoSummary(BaseModel):
    youtube_video_id: str
    video_title: str
    chunk_count: int


class ListVideosResponse(BaseModel):
    chatbots: dict[str, list[VideoSummary]]
    total_chatbots: int
    total_videos: int
    total_chunks: int