import asyncio
import logging
import re
import time

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


async def process_chat(request: ChatProcessRequest) -> ChatProcessResponse:
    query = request.query.strip()

    language = detect_language(query, history=request.conversation_history)

    # Classify intent and rewrite query concurrently — both are independent LLM calls
    intent, final_query = await asyncio.gather(
        get_intent_classifier().classify(query),
        get_query_rewriter().rewrite(query, request.conversation_history, language=language),
    )
    logger.info("step=intent_done intent=%s", intent)
    logger.info("step=rewrite_done query=%r", final_query)

    # Short-circuit for non-content intents
    if intent != "CONTENT_QUESTION":
        answer = get_hardcoded_response(intent, language, request.config.chatbot_name)
        return ChatProcessResponse(answer=answer, sources=[])

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
            query_embedding,
            settings.TOP_K,
            settings.SIMILARITY_THRESHOLD,
        )
        retrieval_ms = int((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        raise VectorStoreError("Retrieval failed") from exc
    logger.info("step=retrieve_done chunks=%d retrieval_ms=%d", len(chunks), retrieval_ms)

    if not chunks:
        template = _FALLBACK_TEMPLATES.get(language, _FALLBACK_TEMPLATES["en"])
        return ChatProcessResponse(
            answer=template.format(chatbot_name=request.config.chatbot_name),
            sources=[],
        )

    messages = build_messages(final_query, chunks, request.conversation_history, request.config, language)

    try:
        if request.first_message:
            # Generate answer and session title concurrently
            (answer, llm_ms), session_title = await asyncio.gather(
                get_llm_client().generate(messages),
                get_title_generator().generate(final_query),
            )
        else:
            answer, llm_ms = await get_llm_client().generate(messages)
            session_title = None
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

    return ChatProcessResponse(answer=answer, session_title=session_title, sources=sources)
