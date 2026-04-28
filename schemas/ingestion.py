from typing import Literal
from pydantic import BaseModel


# ── Inbound (FastAPI receives from Spring Boot) ────────────────────────────

class VideoInput(BaseModel):
    youtube_video_id: str
    video_title: str


class IndexVideosRequest(BaseModel):
    chatbot_id: str
    videos: list[VideoInput]


# ── Outbound: 202 response (FastAPI → Spring Boot, immediately) ────────────

class IndexVideosAcceptedResponse(BaseModel):
    status: Literal["ACCEPTED"]
    chatbot_id: str
    total: int                   # how many videos were queued


# ── Outbound: per-video callback (FastAPI → Spring Boot, as each finishes) ─

class VideoIndexedCallback(BaseModel):
    youtube_video_id: str
    status: Literal["COMPLETED", "FAILED"]
    error: str | None = None             # None on success


class DeleteVideoRequest(BaseModel):
    youtube_video_id: str


class DeleteVideoResponse(BaseModel):
    youtube_video_id: str
    deleted_chunks: int