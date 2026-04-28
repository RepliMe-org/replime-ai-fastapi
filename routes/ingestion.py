from fastapi import APIRouter, BackgroundTasks, Body, Depends, status

from core.dependencies import verify_internal_token
from rag.vector_store import get_vector_store
from schemas.ingestion import (
    DeleteVideoRequest,
    DeleteVideoResponse,
    IndexVideosAcceptedResponse,
    IndexVideosRequest,
)
from services.ingestion_service import run_ingestion

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post(
    "/ingest/videos",
    response_model=IndexVideosAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_videos(
    request: IndexVideosRequest,
    background_tasks: BackgroundTasks,
) -> IndexVideosAcceptedResponse:
    for video in request.videos:
        background_tasks.add_task(
            run_ingestion,
            request.chatbot_id,
            video.youtube_video_id,
            video.video_title,
        )
    return IndexVideosAcceptedResponse(
        status="ACCEPTED",
        chatbot_id=request.chatbot_id,
        total=len(request.videos),
    )


@router.delete("/delete/video", response_model=DeleteVideoResponse)
def delete_video(
    request: DeleteVideoRequest = Body(...),
) -> DeleteVideoResponse:
    count = get_vector_store().delete_by_video_id(request.chatbot_id, request.youtube_video_id)
    return DeleteVideoResponse(youtube_video_id=request.youtube_video_id, deleted_chunks=count)
