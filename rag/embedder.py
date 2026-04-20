import asyncio
import logging
import time

from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s'…", self._model_name)
            t0 = time.perf_counter()
            self._model = SentenceTransformer(self._model_name)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Embedding model loaded in %.1f ms", elapsed)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode, texts)

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings.EMBEDDING_MODEL_ID)
    return _embedder