import logging
import time

from core.config import settings
from core.exceptions import EmbeddingError, LLMError, VectorStoreError
from rag.embedder import get_embedder
from rag.language_detector import detect_language
from rag.llm_client import get_llm_client
from rag.prompt_builder import build_messages
from rag.query_rewriter import get_query_rewriter
from rag.vector_store import get_vector_store
from schemas.chat import ChatProcessRequest, ChatProcessResponse, Source

logger = logging.getLogger(__name__)

_FALLBACK_TEMPLATES = {
    "en": "I don't have information about that in {chatbot_name}'s content.",
    "ar": "لا أملك معلومات حول ذلك في محتوى {chatbot_name}.",
}


async def process_chat(request: ChatProcessRequest) -> ChatProcessResponse:
    query = request.query.strip()

    language = detect_language(query, fallback=request.config.default_language)

    final_query = await get_query_rewriter().rewrite(query, request.conversation_history)
    logger.info("step=rewrite_done query=%r", final_query)

    try:
        query_embedding = await get_embedder().embed_one(final_query)
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
        answer, llm_ms = await get_llm_client().generate(messages)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        raise LLMError("LLM generation failed") from exc
    logger.info("step=generate_done llm_ms=%d", llm_ms)

    sources = []
    for chunk in chunks:
        ts = chunk["timestamp_seconds"]
        timestamp_seconds = ts if ts is not None and ts >= 0 else 0
        sources.append(
            Source(
                video_id=chunk["youtube_video_id"],
                video_title=chunk["video_title"],
                chunk_text=chunk["chunk_text"],
                youtube_url=f"https://youtube.com/watch?v={chunk['youtube_video_id']}&t={timestamp_seconds}s",
                timestamp_seconds=timestamp_seconds,
            )
        )

    return ChatProcessResponse(answer=answer, sources=sources)
