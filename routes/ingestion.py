from fastapi import APIRouter, BackgroundTasks, Depends, status, Body

from core.dependencies import verify_internal_token
from rag.vector_store import get_vector_store
from schemas.ingestion import (
    DeleteVideoRequest,
    DeleteVideoResponse,
    IndexVideoRequest,
    IndexVideoResponse,
)
from services.ingestion_service import run_ingestion

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.post(
    "/internal/videos/{video_id}/index",
    response_model=IndexVideoResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_video(
    video_id: str,
    request: IndexVideoRequest,
    background_tasks: BackgroundTasks,
) -> IndexVideoResponse:
    background_tasks.add_task(
        run_ingestion,
        video_id,
        request.chatbot_id,
        request.youtube_video_id,
        request.video_title,
    )
    return IndexVideoResponse(status="accepted", video_id=video_id)


@router.delete(
    "/internal/videos/{video_id}",
    response_model=DeleteVideoResponse,
)
def delete_video(
    video_id: str,
    request: DeleteVideoRequest = Body(...),
) -> DeleteVideoResponse:
    count = get_vector_store().delete_by_video_id(request.chatbot_id, video_id)
    return DeleteVideoResponse(video_id=video_id, deleted_chunks=count)
