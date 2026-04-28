import logging
import time

from rag.embedder import get_embedder
from rag.llm_client import get_llm_client
from rag.prompt_builder import build_prompt
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

    llm = get_llm_client()
    final_query = await get_query_rewriter().rewrite(query, request.conversation_history)
    rewritten_query = final_query
    logger.info("step=rewrite_done session_id=%s query=%r", request.session_id, final_query)

    query_embedding = await get_embedder().embed_one(final_query)
    logger.info("step=embed_done session_id=%s", request.session_id)

    t0 = time.perf_counter()
    chunks = get_vector_store().search(
        request.chatbot_id,
        query_embedding,
        request.config.top_k,
        request.config.similarity_threshold,
    )
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("step=retrieve_done session_id=%s query=%r chunks=%d retrieval_ms=%d", request.session_id, final_query, len(chunks), retrieval_ms)

    if not chunks:
        logger.info("step=fallback session_id=%s language=%s", request.session_id, request.language)
        template = _FALLBACK_TEMPLATES.get(request.language, _FALLBACK_TEMPLATES["en"])
        return ChatProcessResponse(
            answer=template.format(chatbot_name=request.config.chatbot_name),
            sources=[],
            retrieval_ms=retrieval_ms,
            llm_ms=0,
            rewritten_query=rewritten_query,
        )

    messages = build_prompt(final_query, chunks, request.conversation_history, request.config)
    answer, llm_ms = await llm.generate(messages)
    logger.info("step=generate_done session_id=%s llm_ms=%d", request.session_id, llm_ms)

    sources = [
        Source(
            video_title=chunk["video_title"],
            chunk_text=chunk["chunk_text"],
            youtube_url=(
                f"https://youtube.com/watch?v={chunk['youtube_video_id']}"
                f"&t={chunk['timestamp_seconds']}s"
            ),
            timestamp_seconds=chunk["timestamp_seconds"],
            similarity_score=chunk["similarity_score"],
        )
        for chunk in chunks
    ]

    return ChatProcessResponse(
        answer=answer,
        sources=sources,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        rewritten_query=rewritten_query,
    )
