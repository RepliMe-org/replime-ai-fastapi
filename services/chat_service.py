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
    Return cited chunks and the answer with citations renumbered to match the sources array.
    e.g. if answer cites [Source 2] and [Source 7], they become [Source 1] and [Source 2].
    Falls back to all chunks (no renumbering) if the answer contains no citations.
    """
    original_indices = sorted({int(m) for m in re.findall(r"Source\s+(\d+)", answer)})
    valid_indices = [i for i in original_indices if 1 <= i <= len(chunks)]

    if not valid_indices:
        return chunks, answer

    cited = [chunks[i - 1] for i in valid_indices]

    # Build renumbering map: original index → new 1-based position
    remap = {orig: new for new, orig in enumerate(valid_indices, start=1)}

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        return f"Source {remap[n]}" if n in remap else match.group(0)

    renumbered_answer = re.sub(r"Source\s+(\d+)", _replace, answer)
    return cited, renumbered_answer


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
    sources = []
    for chunk in cited_chunks:
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
