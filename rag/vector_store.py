import logging

import chromadb
from chromadb.errors import ChromaError
from core.exceptions import VectorStoreError

from core.config import settings

logger = logging.getLogger(__name__)



class VectorStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._client: chromadb.PersistentClient | None = None

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._path)
        return self._client

    def _get_collection(self, chatbot_id: str) -> chromadb.Collection:
        client = self._get_client()
        return client.get_or_create_collection(
            name=f"chatbot_{chatbot_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chatbot_id: str,
        youtube_video_id: str,
        video_title: str,
        chunks: list[str],
        embeddings: list[list[float]],
        timestamps: list[int | None],
    ) -> None:
        try:
            collection = self._get_collection(chatbot_id)
            ids = [f"chunk_{youtube_video_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "youtube_video_id": youtube_video_id,
                    "video_title": video_title,
                    "timestamp_seconds": timestamps[i] if timestamps[i] is not None else -1,
                }
                for i in range(len(chunks))
            ]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            logger.info("Upserted %d chunks for youtube_video_id=%s", len(chunks), youtube_video_id)
        except ChromaError as exc:
            raise VectorStoreError(f"upsert_chunks failed: {exc}") from exc

    def search(
        self,
        chatbot_id: str,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float,
    ) -> list[dict]:
        try:
            collection = self._get_collection(chatbot_id)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            for doc, meta, dist in zip(documents, metadatas, distances):
                # cosine space: distance in [0,2], similarity = 1 - distance/2
                similarity = 1.0 - dist / 2.0
                if similarity < similarity_threshold:
                    continue
                output.append(
                    {
                        "chunk_text": doc,
                        "youtube_video_id": meta["youtube_video_id"],
                        "video_title": meta["video_title"],
                        "timestamp_seconds": meta["timestamp_seconds"],
                        "similarity_score": similarity,
                    }
                )
            return output
        except ChromaError as exc:
            raise VectorStoreError(f"search failed: {exc}") from exc

    def delete_by_video_id(self, chatbot_id: str, youtube_video_id: str) -> int:
        try:
            collection = self._get_collection(chatbot_id)
            results = collection.get(
                where={"youtube_video_id": youtube_video_id},
                include=[],
            )
            ids = results["ids"]
            if ids:
                collection.delete(ids=ids)
            logger.info("Deleted %d chunks for youtube_video_id=%s", len(ids), youtube_video_id)
            return len(ids)
        except ChromaError as exc:
            raise VectorStoreError(f"delete_by_video_id failed: {exc}") from exc


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(settings.CHROMA_PATH)
    return _vector_store
