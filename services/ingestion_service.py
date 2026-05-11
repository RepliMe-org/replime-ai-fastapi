import logging

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from rag.chunker import chunk_transcript
from rag.embedder import get_embedder
from rag.language_detector import detect_language
from rag.transcript_loader import load_transcript
from rag.vector_store import get_vector_store
from schemas.ingestion import VideoIndexedCallback

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=10.0)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _is_retryable_callback_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable_callback_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    reraise=True,
)
async def _do_callback(callback: VideoIndexedCallback) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/internal/update-video-status/{callback.youtube_video_id}"
    response = await get_http_client().patch(
        url,
        json={"status": callback.status, "error": callback.error},
        headers={"X-Internal-Token": settings.INTERNAL_TOKEN},
    )
    response.raise_for_status()


async def send_callback(callback: VideoIndexedCallback) -> None:
    try:
        await _do_callback(callback)
    except Exception as exc:
        logger.exception(
            "send_callback failed after retries: youtube_video_id=%s error=%s",
            callback.youtube_video_id,
            exc,
        )


def _detect_transcript_language(segments: list[dict]) -> str:
    sample = " ".join(seg["text"] for seg in segments[:20])
    return detect_language(sample)


async def run_ingestion(
    chatbot_id: str,
    youtube_video_id: str,
    video_title: str | None = None,
) -> None:
    title = video_title or youtube_video_id
    try:
        logger.info("step=load_transcript youtube_video_id=%s", youtube_video_id)
        segments = load_transcript(youtube_video_id)

        language = _detect_transcript_language(segments)
        logger.info("step=detect_language language=%s youtube_video_id=%s", language, youtube_video_id)

        logger.info("step=chunk_transcript youtube_video_id=%s", youtube_video_id)
        chunks = chunk_transcript(segments, language=language)

        logger.info("step=embed youtube_video_id=%s", youtube_video_id)
        embeddings = await get_embedder().embed_documents([c["text"] for c in chunks])

        logger.info("step=upsert youtube_video_id=%s", youtube_video_id)
        get_vector_store().upsert_chunks(
            chatbot_id,
            youtube_video_id,
            title,
            [c["text"] for c in chunks],
            embeddings,
            [c.get("timestamp_seconds") for c in chunks],
        )

        logger.info("step=done youtube_video_id=%s", youtube_video_id)
        await send_callback(VideoIndexedCallback(
            youtube_video_id=youtube_video_id,
            status="COMPLETED",
        ))

    except Exception as exc:
        logger.exception("ingestion failed youtube_video_id=%s", youtube_video_id)
        await send_callback(VideoIndexedCallback(
            youtube_video_id=youtube_video_id,
            status="FAILED",
            error=str(exc),
        ))
