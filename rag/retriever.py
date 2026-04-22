import logging

from rag.embedder import Embedder, get_embedder
from rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_DEFAULT_SIMILARITY_THRESHOLD = 0.4


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def retrieve(
        self,
        chatbot_id: str,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        query_embedding = await self._embedder.embed_one(query)
        results = self._vector_store.search(
            chatbot_id=chatbot_id,
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        logger.info(
            "Retrieved %d chunks for chatbot_id=%s",
            len(results),
            chatbot_id,
            extra={"chatbot_id": chatbot_id, "query_preview": query[:80]},
        )
        return results


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_embedder(), get_vector_store())
    return _retriever
