"""Channel-description maintenance.

The channel description (what topics a creator's channel covers) is regenerated
from the chatbot's *current* Qdrant chunks whenever the content set changes
(video ingested or deleted). Qdrant is the source of truth, so the description
is convergent: deletes self-heal and there is no incremental-merge drift.

Updates are serialized per chatbot (Redis lock, in-process fallback) so
concurrent ingest/delete tasks can't clobber each other, and the result is
reported to Spring Boot via a dedicated callback.
"""
import asyncio
import contextlib
import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from infrastructure.http_client import get_http_client, is_retryable_http_error
from infrastructure.redis_client import get_redis
from rag.llm.description_generator import get_description_generator
from rag.retrieval.vector_store import get_vector_store
from services.corpus_language_service import get_corpus_language

logger = logging.getLogger(__name__)

# In-process fallback locks, used only when Redis is unavailable. One per chatbot.
_local_locks: dict[str, asyncio.Lock] = {}

# Strong refs to detached regen tasks so the event loop doesn't GC them mid-flight.
_background_refreshes: set[asyncio.Task] = set()


def _local_lock(chatbot_id: str) -> asyncio.Lock:
    lock = _local_locks.get(chatbot_id)
    if lock is None:
        lock = asyncio.Lock()
        _local_locks[chatbot_id] = lock
    return lock


async def _keep_lock_alive(redis_lock, chatbot_id: str) -> None:
    """Re-extend a held Redis lock on a heartbeat so a slow-but-live regeneration
    keeps its lease. The description LLM call can run for minutes (NVIDIA batch),
    far longer than DESCRIPTION_LOCK_TTL_SECONDS; without this the lease would
    expire mid-flight and a concurrent regen could grab the "free" lock. Runs until
    cancelled — if the process dies the lease still expires, letting another worker
    recover.
    """
    interval = max(1, settings.DESCRIPTION_LOCK_TTL_SECONDS // 3)
    while True:
        await asyncio.sleep(interval)
        try:
            await redis_lock.reacquire()
        except Exception as exc:
            logger.warning("description lock re-extend failed chatbot_id=%s: %s", chatbot_id, exc)
            return


def spawn_channel_description_refresh(chatbot_id: str) -> None:
    """Fire-and-forget channel-description regeneration.

    Lets the ingestion pipeline report a video COMPLETED as soon as its chunks are
    indexed, instead of blocking the webhook behind the slow, best-effort
    description LLM call. Serialized per chatbot by refresh_channel_description's
    own lock; it reports its own result to Spring Boot.
    """
    task = asyncio.create_task(refresh_channel_description(chatbot_id))
    _background_refreshes.add(task)
    task.add_done_callback(_background_refreshes.discard)


# ── Spring Boot callback ───────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception(is_retryable_http_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    reraise=True,
)
async def _do_description_callback(chatbot_id: str, payload: dict) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/internal/chatbots/{chatbot_id}/ai-description"
    response = await get_http_client().patch(
        url,
        json=payload,
        headers={"X-INTERNAL-TOKEN": settings.X_INTERNAL_TOKEN},
    )
    response.raise_for_status()


async def send_description_callback(chatbot_id: str, description: str | None) -> None:
    """PATCH the regenerated channel description to Spring Boot. Logs, never raises.

    A null ``description`` signals 'no content' (e.g. the last video was deleted);
    Spring Boot decides whether to fall back to the creator's seed.
    """
    try:
        await _do_description_callback(chatbot_id, {"description": description})
    except Exception as exc:
        logger.exception("send_description_callback failed chatbot_id=%s: %s", chatbot_id, exc)


# ── Regeneration ───────────────────────────────────────────────────────────────

async def _regenerate_and_report(chatbot_id: str) -> str | None:
    # Re-read the sample *inside* the lock so it reflects committed Qdrant state.
    sample = await asyncio.to_thread(
        get_vector_store().sample_chunks,
        chatbot_id,
        settings.DESCRIPTION_SAMPLE_PER_VIDEO,
        settings.DESCRIPTION_SAMPLE_MAX_CHARS,
    )
    if not sample:
        # No chunks remain (e.g. last video deleted) — clear the description.
        logger.info("no chunks for chatbot_id=%s — clearing description", chatbot_id)
        await send_description_callback(chatbot_id, None)
        return None

    # Output language *and* model both follow the chatbot's dominant corpus language
    # (authoritative, from stored content_language) rather than langdetect on the
    # excerpt sample.
    language = await get_corpus_language(chatbot_id)
    model_spec = settings.model_for(settings.DESCRIPTION_MODEL, settings.DESCRIPTION_MODEL_AR, language)
    generated = await get_description_generator(model_spec).regenerate(sample, language)
    description = generated or None
    logger.info(
        "description regenerated chatbot_id=%s length=%d", chatbot_id, len(description or "")
    )
    await send_description_callback(chatbot_id, description)
    return description


async def refresh_channel_description(chatbot_id: str) -> str | None:
    """Regenerate the channel description from current Qdrant state and report it.

    Serialized per chatbot. Best-effort: logs and returns None on failure rather
    than raising, since callers treat description maintenance as non-critical (the
    video is already indexed/deleted regardless).
    """
    redis_lock = None
    local = None
    try:
        redis = await get_redis()
        redis_lock = redis.lock(
            f"desc-lock:{chatbot_id}",
            timeout=settings.DESCRIPTION_LOCK_TTL_SECONDS,
            blocking_timeout=settings.DESCRIPTION_LOCK_WAIT_SECONDS,
        )
        if not await redis_lock.acquire():
            logger.warning("description lock timeout chatbot_id=%s — skipping refresh", chatbot_id)
            return None
    except Exception as exc:
        logger.warning(
            "Redis lock unavailable chatbot_id=%s, using in-process lock: %s", chatbot_id, exc
        )
        redis_lock = None
        local = _local_lock(chatbot_id)
        await local.acquire()

    # Heartbeat the Redis lease so the lock survives a slow regen (see _keep_lock_alive).
    watchdog = asyncio.create_task(_keep_lock_alive(redis_lock, chatbot_id)) if redis_lock else None
    try:
        return await _regenerate_and_report(chatbot_id)
    except Exception as exc:
        logger.warning("description refresh failed chatbot_id=%s: %s", chatbot_id, exc)
        return None
    finally:
        if watchdog is not None:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
        if redis_lock is not None:
            with contextlib.suppress(Exception):
                await redis_lock.release()
        if local is not None:
            local.release()
