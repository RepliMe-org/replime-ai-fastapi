import logging

import httpx

from core.config import settings
from rag.chunker import chunk_transcript
from rag.embedder import get_embedder
from rag.transcript_loader import load_transcript
from rag.vector_store import get_vector_store
from schemas.ingestion import VideoIndexedCallback

logger = logging.getLogger(__name__)


async def send_callback(callback: VideoIndexedCallback) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/internal/update-video-status/{callback.youtube_video_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(url, json={"status": callback.status, "error": callback.error})
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.exception(
            "send_callback failed",
            extra={"youtube_video_id": callback.youtube_video_id, "error": str(e)},
        )


async def run_ingestion(
    chatbot_id: str,
    youtube_video_id: str,
    video_title: str | None = None,
) -> None:
    title = video_title or youtube_video_id
    try:
        logger.info("step=load_transcript", extra={"youtube_video_id": youtube_video_id})
        segments = load_transcript(youtube_video_id)

        logger.info("step=chunk_transcript", extra={"youtube_video_id": youtube_video_id})
        chunks = chunk_transcript(segments)

        logger.info("step=embed", extra={"youtube_video_id": youtube_video_id})
        embeddings = await get_embedder().embed([c["text"] for c in chunks])

        logger.info("step=upsert", extra={"youtube_video_id": youtube_video_id})
        get_vector_store().upsert_chunks(
            chatbot_id,
            youtube_video_id,
            title,
            [c["text"] for c in chunks],
            embeddings,
            [c.get("timestamp_seconds") for c in chunks],
        )

        logger.info("step=done", extra={"youtube_video_id": youtube_video_id})
        await send_callback(VideoIndexedCallback(
            youtube_video_id=youtube_video_id,
            status="COMPLETED",
        ))

    except Exception as e:
        logger.exception("ingestion failed", extra={"youtube_video_id": youtube_video_id})
        await send_callback(VideoIndexedCallback(
            youtube_video_id=youtube_video_id,
            status="FAILED",
            error=str(e),
        ))
