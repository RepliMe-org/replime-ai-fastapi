import logging
import uuid

from qdrant_client import models

from core.config import settings
from core.exceptions import VectorStoreError
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Profiles are pure-payload points (no semantic vector needed — we fetch by id).
# Qdrant still requires a vector, so we attach a tiny constant 1-dim dummy vector.
_DUMMY_VECTOR_NAME = "dummy"
_DUMMY_VECTOR = [0.0]
_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _profile_id(chatbot_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"profile_{chatbot_id}"))


class ProfileStore:
    """Stores one accumulating text profile per chatbot in the chatbot_meta collection.

    The profile is an AI-generated description of what topics the channel's content
    covers. It's used by the intent classifier to distinguish adjacent-but-uncovered
    questions (CONTENT_GAP) from completely off-topic ones (OUT_OF_SCOPE).
    """

    def __init__(self, collection: str) -> None:
        self._collection = collection
        self._ensured = False

    def _client(self):
        # Reuse the singleton QdrantClient managed by the main vector store.
        return get_vector_store()._get_client()

    def _ensure_collection(self) -> None:
        if self._ensured:
            return
        client = self._client()
        if not client.collection_exists(self._collection):
            client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    _DUMMY_VECTOR_NAME: models.VectorParams(
                        size=1, distance=models.Distance.COSINE
                    )
                },
            )
            logger.info("Created Qdrant collection '%s'", self._collection)
        self._ensured = True

    def get_profile(self, chatbot_id: str) -> str | None:
        try:
            self._ensure_collection()
            points = self._client().retrieve(
                collection_name=self._collection,
                ids=[_profile_id(chatbot_id)],
                with_payload=True,
            )
            if not points:
                return None
            return (points[0].payload or {}).get("profile_text")
        except Exception as exc:
            # Profile is an enhancement, not critical — log and degrade gracefully.
            logger.warning("get_profile failed for chatbot_id=%s: %s", chatbot_id, exc)
            return None

    def upsert_profile(self, chatbot_id: str, profile_text: str) -> None:
        try:
            self._ensure_collection()
            self._client().upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=_profile_id(chatbot_id),
                        vector={_DUMMY_VECTOR_NAME: _DUMMY_VECTOR},
                        payload={
                            "chatbot_id": chatbot_id,
                            "profile_text": profile_text,
                        },
                    )
                ],
            )
            logger.info("Upserted channel profile for chatbot_id=%s", chatbot_id)
        except Exception as exc:
            raise VectorStoreError(f"upsert_profile failed: {exc}") from exc


_profile_store: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    global _profile_store
    if _profile_store is None:
        _profile_store = ProfileStore(settings.CHATBOT_META_COLLECTION)
    return _profile_store
