import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from core.exceptions import (
    NonRetryableIngestionError,
    RetryableIngestionError,
    TranscriptError,
    TranscriptRateLimitError,
    VectorStoreError,
)
from rag.chunker import chunk_transcript
from rag.embedder import get_embedder
from rag.language_detector import detect_language
from rag.transcript_loader import load_transcript
from rag.vector_store import get_vector_store
from services.http_client import get_http_client, is_retryable_http_error

logger = logging.getLogger(__name__)

_STAGE_TRANSCRIPT = "TRANSCRIPT_EXTRACTION"
_STAGE_CHUNKING = "CHUNKING"
_STAGE_EMBEDDING = "EMBEDDING_GENERATION"
_STAGE_INDEXING = "VECTOR_INDEXING"


# ── Webhook callback ───────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception(is_retryable_http_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    reraise=True,
)
async def _do_callback(youtube_video_id: str, payload: dict) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/internal/update-video-status/{youtube_video_id}"
    response = await get_http_client().patch(
        url,
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_TOKEN},
    )
    response.raise_for_status()


async def send_ingestion_callback(youtube_video_id: str, payload: dict) -> None:
    """Post a status webhook to Spring Boot. Logs but does not raise on failure."""
    try:
        await _do_callback(youtube_video_id, payload)
    except Exception as exc:
        logger.exception(
            "send_ingestion_callback failed after retries: youtube_video_id=%s error=%s",
            youtube_video_id,
            exc,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_ingestion_pipeline(
    chatbot_id: str,
    youtube_video_id: str,
    video_title: str | None,
) -> None:
    """Run all ingestion stages in order.

    Raises NonRetryableIngestionError for permanent failures (no transcript,
    private video) and RetryableIngestionError for transient ones. The caller
    is responsible for sending the webhook callback.
    """
    title = video_title or youtube_video_id

    # Stage 1 — transcript extraction
    try:
        logger.info("stage=%s youtube_video_id=%s", _STAGE_TRANSCRIPT, youtube_video_id)
        segments = load_transcript(youtube_video_id)
        language = detect_language(" ".join(seg["text"] for seg in segments[:20]))
        logger.info("language=%s youtube_video_id=%s", language, youtube_video_id)
    except TranscriptRateLimitError as exc:
        raise RetryableIngestionError(_STAGE_TRANSCRIPT, str(exc)) from exc
    except TranscriptError as exc:
        raise NonRetryableIngestionError(_STAGE_TRANSCRIPT, str(exc)) from exc
    except Exception as exc:
        raise RetryableIngestionError(_STAGE_TRANSCRIPT, f"Unexpected error: {exc}") from exc

    # Stage 2 — chunking
    try:
        logger.info("stage=%s youtube_video_id=%s", _STAGE_CHUNKING, youtube_video_id)
        chunks = chunk_transcript(segments, language=language)
    except Exception as exc:
        raise RetryableIngestionError(_STAGE_CHUNKING, f"Chunking failed: {exc}") from exc

    # Stage 3 — embedding
    try:
        logger.info("stage=%s youtube_video_id=%s", _STAGE_EMBEDDING, youtube_video_id)
        embeddings = await get_embedder().embed_documents([c["text"] for c in chunks])
    except Exception as exc:
        raise RetryableIngestionError(_STAGE_EMBEDDING, f"Embedding failed: {exc}") from exc

    # Stage 4 — vector indexing
    try:
        logger.info("stage=%s youtube_video_id=%s", _STAGE_INDEXING, youtube_video_id)
        get_vector_store().upsert_chunks(
            chatbot_id,
            youtube_video_id,
            title,
            [c["text"] for c in chunks],
            embeddings,
            [c.get("timestamp_seconds") for c in chunks],
        )
    except VectorStoreError as exc:
        raise RetryableIngestionError(_STAGE_INDEXING, str(exc)) from exc
    except Exception as exc:
        raise RetryableIngestionError(_STAGE_INDEXING, f"Unexpected indexing error: {exc}") from exc

    logger.info("stage=done youtube_video_id=%s", youtube_video_id)


# ── HTTP-triggered ingestion (POST /ai/ingest/videos) ─────────────────────────

async def run_ingestion(
    chatbot_id: str,
    youtube_video_id: str,
    video_title: str | None = None,
) -> None:
    """Single-attempt ingestion triggered via the HTTP endpoint (attempt=1).

    Runs the pipeline and sends the webhook callback regardless of outcome.
    """
    try:
        await run_ingestion_pipeline(chatbot_id, youtube_video_id, video_title)
        await send_ingestion_callback(youtube_video_id, {
            "status": "COMPLETED",
            "attemptsMade": 1,
        })
    except NonRetryableIngestionError as exc:
        logger.error(
            "ingestion permanent failure youtube_video_id=%s stage=%s reason=%s",
            youtube_video_id, exc.stage, exc.reason,
        )
        await send_ingestion_callback(youtube_video_id, {
            "status": "FAILED",
            "failedStage": exc.stage,
            "failureReason": exc.reason,
            "attemptsMade": 1,
            "retryable": False,
        })
    except RetryableIngestionError as exc:
        logger.warning(
            "ingestion transient failure youtube_video_id=%s stage=%s reason=%s",
            youtube_video_id, exc.stage, exc.reason,
        )
        await send_ingestion_callback(youtube_video_id, {
            "status": "FAILED",
            "failedStage": exc.stage,
            "failureReason": exc.reason,
            "attemptsMade": 1,
            "retryable": True,
        })
    except Exception as exc:
        logger.exception("ingestion failed unexpectedly youtube_video_id=%s", youtube_video_id)
        await send_ingestion_callback(youtube_video_id, {
            "status": "FAILED",
            "failureReason": str(exc),
            "attemptsMade": 1,
            "retryable": True,
        })
