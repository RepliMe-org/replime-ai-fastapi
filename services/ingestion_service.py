import logging

import httpx

from core.config import settings
from rag.chunker import chunk_transcript
from rag.embedder import get_embedder
from rag.transcript_loader import load_transcript
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


async def notify_status(video_id: str, status: str, error: str | None = None) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/api/v1/internal/videos/{video_id}/status"
    payload: dict = {"sync_status": status, "error_message": error}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.put(url, json=payload)
            response.raise_for_status()
    except Exception:
        logger.exception("notify_status failed", extra={"video_id": video_id, "status": status})


async def run_ingestion(
    video_id: str,
    chatbot_id: str,
    youtube_video_id: str,
    video_title: str | None = None,
) -> None:
    title = video_title or youtube_video_id
    try:
        logger.info("step=load_transcript", extra={"video_id": video_id})
        segments = load_transcript(youtube_video_id)

        logger.info("step=chunk_transcript", extra={"video_id": video_id})
        chunks = chunk_transcript(segments)

        logger.info("step=embed", extra={"video_id": video_id})
        embeddings = await get_embedder().embed([c["text"] for c in chunks])

        logger.info("step=upsert", extra={"video_id": video_id})
        get_vector_store().upsert_chunks(
            chatbot_id,
            video_id,
            youtube_video_id,
            title,
            [c["text"] for c in chunks],
            embeddings,
            [c.get("timestamp_seconds") for c in chunks],
        )

        logger.info("step=done", extra={"video_id": video_id})
        await notify_status(video_id, "COMPLETED")

    except Exception as e:
        logger.exception("ingestion failed", extra={"video_id": video_id})
        await notify_status(video_id, "FAILED", str(e))
