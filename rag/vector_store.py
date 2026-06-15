import logging
import uuid

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

from core.config import settings
from core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

# intfloat/multilingual-e5-large produces 1024-dim dense vectors.
_DENSE_SIZE = 1024
_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"

# Stable namespace so point ids are deterministic across re-ingestion (idempotent upserts).
_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# How many candidates each branch contributes before RRF fusion narrows to top_k.
_PREFETCH_LIMIT = 20


def _point_id(chatbot_id: str, youtube_video_id: str, index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{chatbot_id}_{youtube_video_id}_{index}"))


class VectorStore:
    def __init__(self, url: str, api_key: str, collection: str) -> None:
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._client: QdrantClient | None = None
        self._sparse_model: SparseTextEmbedding | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self._url, api_key=self._api_key or None)
            self._ensure_collection(self._client)
        return self._client

    def _ensure_collection(self, client: QdrantClient) -> None:
        if client.collection_exists(self._collection):
            return
        client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE_NAME: models.VectorParams(
                    size=_DENSE_SIZE, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                _SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        # Index the payload field we filter on for fast multi-tenant lookups.
        client.create_payload_index(
            collection_name=self._collection,
            field_name="chatbot_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection '%s'", self._collection)

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            logger.info("Loading sparse model '%s'…", settings.SPARSE_MODEL_ID)
            self._sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_ID)
        return self._sparse_model

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
            client = self._get_client()
            sparse_embeddings = list(self._get_sparse_model().embed(chunks))
            points = []
            for i, (chunk, dense, sparse) in enumerate(
                zip(chunks, embeddings, sparse_embeddings)
            ):
                points.append(
                    models.PointStruct(
                        id=_point_id(chatbot_id, youtube_video_id, i),
                        vector={
                            _DENSE_NAME: dense,
                            _SPARSE_NAME: models.SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                        },
                        payload={
                            "chatbot_id": chatbot_id,
                            "youtube_video_id": youtube_video_id,
                            "video_title": video_title,
                            "timestamp_seconds": timestamps[i]
                            if timestamps[i] is not None
                            else -1,
                            "chunk_text": chunk,
                        },
                    )
                )
            client.upsert(collection_name=self._collection, points=points)
            logger.info(
                "Upserted %d chunks for youtube_video_id=%s", len(points), youtube_video_id
            )
        except Exception as exc:
            raise VectorStoreError(f"upsert_chunks failed: {exc}") from exc

    def search(
        self,
        chatbot_id: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float,
    ) -> list[dict]:
        try:
            client = self._get_client()
            sparse = next(iter(self._get_sparse_model().query_embed(query_text)))
            chatbot_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chatbot_id",
                        match=models.MatchValue(value=chatbot_id),
                    )
                ]
            )
            response = client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=query_embedding,
                        using=_DENSE_NAME,
                        filter=chatbot_filter,
                        score_threshold=similarity_threshold,
                        limit=_PREFETCH_LIMIT,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                        using=_SPARSE_NAME,
                        filter=chatbot_filter,
                        limit=_PREFETCH_LIMIT,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            output = []
            for point in response.points:
                payload = point.payload or {}
                output.append(
                    {
                        "chunk_text": payload.get("chunk_text", ""),
                        "youtube_video_id": payload.get("youtube_video_id"),
                        "video_title": payload.get("video_title"),
                        "timestamp_seconds": payload.get("timestamp_seconds"),
                        "similarity_score": point.score,
                    }
                )
            return output
        except Exception as exc:
            raise VectorStoreError(f"search failed: {exc}") from exc

    def delete_by_video_id(self, chatbot_id: str, youtube_video_id: str) -> int:
        try:
            client = self._get_client()
            video_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chatbot_id", match=models.MatchValue(value=chatbot_id)
                    ),
                    models.FieldCondition(
                        key="youtube_video_id",
                        match=models.MatchValue(value=youtube_video_id),
                    ),
                ]
            )
            count = client.count(
                collection_name=self._collection, count_filter=video_filter
            ).count
            if count:
                client.delete(
                    collection_name=self._collection,
                    points_selector=models.FilterSelector(filter=video_filter),
                )
            logger.info(
                "Deleted %d chunks for youtube_video_id=%s", count, youtube_video_id
            )
            return count
        except Exception as exc:
            raise VectorStoreError(f"delete_by_video_id failed: {exc}") from exc

    def healthcheck(self) -> None:
        self._get_client().get_collections()


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            settings.QDRANT_URL,
            settings.QDRANT_API_KEY,
            settings.QDRANT_COLLECTION,
        )
    return _vector_store
