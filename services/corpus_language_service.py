"""Per-chatbot corpus-language tracking.

Model routing (chat_service) needs to know whether a chatbot's *indexed
content* is Arabic-dominant, not what language the current query is in — a
query in one language can still retrieve chunks in another. The dominant
language is recomputed from Qdrant whenever the content set changes (video
ingested or deleted) and cached in Redis for fast reads at chat time.
"""
import logging

from infrastructure.redis_client import get_redis
from rag.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_REDIS_KEY_PREFIX = "corpus_lang"


def _redis_key(chatbot_id: str) -> str:
    return f"{_REDIS_KEY_PREFIX}:{chatbot_id}"


async def refresh_corpus_language(chatbot_id: str) -> str | None:
    """Recompute the chatbot's dominant corpus language and cache it in Redis.

    Best-effort: recomputation is idempotent (derived fresh from current Qdrant
    state each time), so unlike description regeneration this doesn't need a
    per-chatbot lock — a race between concurrent ingest/delete tasks just means
    the cached value settles on whichever finishes last, which is harmless.
    """
    try:
        language = get_vector_store().dominant_language(chatbot_id)
        redis = await get_redis()
        await redis.set(_redis_key(chatbot_id), language)
        logger.info("corpus language refreshed chatbot_id=%s language=%s", chatbot_id, language)
        return language
    except Exception as exc:
        logger.warning("corpus language refresh failed chatbot_id=%s: %s", chatbot_id, exc)
        return None


async def get_corpus_language(chatbot_id: str) -> str:
    """The chatbot's cached dominant corpus language, defaulting to "en".

    Falls back to a live Qdrant recomputation when Redis has no cached value yet
    (e.g. first request after a Redis flush) — best-effort, never raises.
    """
    try:
        redis = await get_redis()
        cached = await redis.get(_redis_key(chatbot_id))
        if cached in ("ar", "en"):
            return cached
    except Exception as exc:
        logger.warning("corpus language cache read failed chatbot_id=%s: %s", chatbot_id, exc)

    language = await refresh_corpus_language(chatbot_id)
    return language or "en"
