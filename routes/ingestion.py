import asyncio

from fastapi import APIRouter, BackgroundTasks, Body, Depends, status

from core.dependencies import verify_internal_token
from rag.retrieval.vector_store import get_vector_store
from schemas.ingestion import (
    DeleteVideoRequest,
    DeleteVideoResponse,
    IndexVideosAcceptedResponse,
    IndexVideosRequest,
    ListVideosResponse,
    VideoSummary,
)
from services.corpus_language_service import refresh_corpus_language
from services.description_service import refresh_channel_description
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
async def delete_video(
    background_tasks: BackgroundTasks,
    request: DeleteVideoRequest = Body(...),
) -> DeleteVideoResponse:
    count = await asyncio.to_thread(
        get_vector_store().delete_by_video_id,
        request.chatbot_id,
        request.youtube_video_id,
    )
    # Content changed — regenerate the channel description from what remains
    # (best-effort, in the background; null when nothing remains). Skip when the
    # delete was a no-op (video had no indexed chunks) to avoid a needless LLM call.
    if count:
        background_tasks.add_task(refresh_channel_description, request.chatbot_id)
        background_tasks.add_task(refresh_corpus_language, request.chatbot_id)
    return DeleteVideoResponse(youtube_video_id=request.youtube_video_id, deleted_chunks=count)


@router.get("/videos", response_model=ListVideosResponse)
def list_videos() -> ListVideosResponse:
    grouped = get_vector_store().list_videos()
    total_videos = sum(len(vids) for vids in grouped.values())
    total_chunks = sum(v["chunk_count"] for vids in grouped.values() for v in vids)
    chatbots = {
        cid: [VideoSummary(**v) for v in vids]
        for cid, vids in grouped.items()
    }
    return ListVideosResponse(
        chatbots=chatbots,
        total_chatbots=len(chatbots),
        total_videos=total_videos,
        total_chunks=total_chunks,
    )
