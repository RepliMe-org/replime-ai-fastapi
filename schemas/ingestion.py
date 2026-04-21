from typing import Literal

from pydantic import BaseModel


class IndexVideoRequest(BaseModel):
    chatbot_id: str
    youtube_video_id: str


class IndexVideoResponse(BaseModel):
    status: Literal["accepted"]
    video_id: str


class DeleteVideoRequest(BaseModel):
    chatbot_id: str


class DeleteVideoResponse(BaseModel):
    video_id: str
    deleted_chunks: int
