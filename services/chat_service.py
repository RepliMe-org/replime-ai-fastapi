import asyncio
import logging
import re
import time

from fastapi import BackgroundTasks

from core.config import settings
from core.exceptions import EmbeddingError, LLMError, VectorStoreError
from rag.embedder import get_embedder
from rag.intent_classifier import get_hardcoded_response, get_intent_classifier
from rag.language_detector import detect_language
from rag.llm_client import get_llm_client
from rag.prompt_builder import build_messages
from rag.query_rewriter import get_query_rewriter
from rag.text_normalizer import normalize_arabic
from rag.title_generator import get_title_generator
from rag.vector_store import get_vector_store
from schemas.chat import ChatProcessRequest, ChatProcessResponse, Source

logger = logging.getLogger(__name__)

_NON_CONTENT_TITLES: dict[str, dict[str, str | None]] = {
    "GREETING":    {"en": "Greeting",          "ar": "تحية ترحيب"},
    "SMALL_TALK":  {"en": "Casual Chat",        "ar": "دردشة عامة"},
    "OUT_OF_SCOPE":{"en": "Off-topic Question", "ar": "سؤال خارج النطاق"},
    "HARMFUL":     {"en": None,                 "ar": None},
}


def _seed_description(config) -> str | None:
    """Build the channel description string from the influencer's description/topics."""
    parts = []
    if config.description:
        parts.append(config.description.strip())
    if config.topics:
        parts.append("Topics: " + ", ".join(config.topics))
    return "\n".join(parts) if parts else None


def _resolve_session_title(intent: str, language: str, generated_title: str | None) -> str | None:
    if intent == "CONTENT_QUESTION":
        return generated_title
    titles = _NON_CONTENT_TITLES.get(intent, {})
    return titles.get(language) or titles.get("en")


_FALLBACK_TEMPLATES = {
    "en": "I don't have information about that in {chatbot_name}'s content.",
    "ar": "لا أملك معلومات حول ذلك في محتوى {chatbot_name}.",
}




def _extract_cited_chunks(answer: str, chunks: list[dict]) -> tuple[list[dict], str]:
    """
    Extract chunks cited in the answer, then strip all [Source N] markers from the answer text.
    Returns (cited_chunks, clean_answer). clean_answer has no [Source N] references.
    """
    # Extract all numbers from bracket citations — handles [1], [1, 3], [1، 3، 5] (Arabic comma)
    bracket_contents = re.findall(r"\[([^\]]+)\]", answer)
    original_indices = sorted({int(n) for content in bracket_contents for n in re.findall(r"\d+", content)})
    valid_indices = [i for i in original_indices if 1 <= i <= len(chunks)]
    cited = [chunks[i - 1] for i in valid_indices]

    # Strip all bracket citations from the answer (Latin and Arabic comma variants)
    clean_answer = re.sub(r"\s*\[[\d\s,،\-–]+\]", "", answer).strip()

    # If the LLM still signals no information, clear cited as backup (English + Arabic)
    _no_info = (
        "i don't have information", "i do not have information", "don't have information about that",
        "لا أملك معلومات", "لا توجد معلومات", "ليس لدي معلومات",
    )
    if any(p in clean_answer.lower() for p in _no_info):
        cited = []

    return cited, clean_answer


async def process_chat(
    request: ChatProcessRequest,
    background_tasks: BackgroundTasks | None = None,
) -> ChatProcessResponse:
    query = request.query.strip()

    language = detect_language(query, history=request.conversation_history)

    # The channel description (sent by Spring Boot in the request config) lets intent
    # classification tell in-domain questions from off-topic ones.
    description = _seed_description(request.config)

    # Classify intent, rewrite query, and (if first message) generate title concurrently
    tasks = [
        get_intent_classifier().classify(query, description=description),
        get_query_rewriter().rewrite(query, request.conversation_history, language=language),
    ]
    if request.first_message:
        tasks.append(get_title_generator().generate(query))

    results = await asyncio.gather(*tasks)
    intent, final_query = results[0], results[1]
    raw_title = results[2] if request.first_message else None
    session_title = _resolve_session_title(intent, language, raw_title) if request.first_message else None

    logger.info("step=intent_done intent=%s", intent)
    logger.info("step=rewrite_done query=%r", final_query)

    # Short-circuit for non-content intents
    if intent != "CONTENT_QUESTION":
        answer = get_hardcoded_response(intent, language, request.config.chatbot_name)
        return ChatProcessResponse(
            answer=answer,
            session_title=session_title,
            sources=[],
            intent=intent,
            message_id=request.message_id,
        )

    # Normalize Arabic query before embedding to match stored normalized chunks
    embed_query = normalize_arabic(final_query) if language == "ar" else final_query

    try:
        query_embedding = await get_embedder().embed_query(embed_query)
    except Exception as exc:
        raise EmbeddingError("Embedding failed") from exc
    logger.info("step=embed_done")

    try:
        t0 = time.perf_counter()
        chunks = get_vector_store().search(
            request.chatbot_id,
            embed_query,
            query_embedding,
            settings.TOP_K,
            settings.SIMILARITY_THRESHOLD,
        )
        retrieval_ms = int((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        raise VectorStoreError("Retrieval failed") from exc
    logger.info("step=retrieve_done chunks=%d retrieval_ms=%d", len(chunks), retrieval_ms)

    if not chunks:
        # An in-domain question we have no content for surfaces later as a content gap
        # (a CONTENT_QUESTION whose answer cited no video).
        template = _FALLBACK_TEMPLATES.get(language, _FALLBACK_TEMPLATES["en"])
        return ChatProcessResponse(
            answer=template.format(chatbot_name=request.config.chatbot_name),
            session_title=session_title,
            sources=[],
            intent=intent,
            message_id=request.message_id,
        )

    messages = build_messages(final_query, chunks, request.conversation_history, request.config, language)

    try:
        answer, llm_ms = await get_llm_client().generate(messages)
    except (LLMError, Exception) as exc:
        logger.error("LLM generation failed: %s", exc)
        raise LLMError("LLM generation failed") from exc
    logger.info("step=generate_done llm_ms=%d session_title=%r", llm_ms, session_title)

    cited_chunks, answer = _extract_cited_chunks(answer, chunks)

    # Deduplicate by video_id, keeping the chunk with the highest similarity score
    best_by_video: dict[str, dict] = {}
    for chunk in cited_chunks:
        vid = chunk["youtube_video_id"]
        if vid not in best_by_video or chunk["similarity_score"] > best_by_video[vid]["similarity_score"]:
            best_by_video[vid] = chunk

    sources = []
    for chunk in best_by_video.values():
        ts = chunk["timestamp_seconds"]
        timestamp_seconds = ts if ts is not None and ts >= 0 else 0
        sources.append(
            Source(
                video_id=chunk["youtube_video_id"],
                video_title=chunk["video_title"],
                youtube_url=f"https://youtube.com/watch?v={chunk['youtube_video_id']}&t={timestamp_seconds}s",
            )
        )

    return ChatProcessResponse(
        answer=answer,
        session_title=session_title,
        sources=sources,
        intent=intent,
        message_id=request.message_id,
    )
