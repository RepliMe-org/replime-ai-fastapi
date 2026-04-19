import chromadb
from core.config import settings
from core.embedding_model import get_embedding_model
from core.exceptions import EmbeddingError, VectorStoreConnectionError
from core.logger import get_logger

logger = get_logger(__name__)


class AIService:
    def health_check(self) -> dict:
        """
        Verify that all core dependencies are healthy:
        - ChromaDB connectivity
        - Embedding model loads
        - Basic vector store connectivity
        """
        health = {
            "status": "ok",
            "service": "ai-fastapi",
            "components": {}
        }

        # Check ChromaDB connection
        try:
            logger.info("Health check: verifying ChromaDB connection")
            client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
            client.heartbeat()
            health["components"]["chroma"] = {"status": "ok"}
            logger.info("ChromaDB: healthy")
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            health["status"] = "degraded"
            health["components"]["chroma"] = {"status": "error", "reason": str(e)}

        # Check embedding model
        try:
            logger.info("Health check: verifying embedding model")
            get_embedding_model()
            health["components"]["embedding_model"] = {"status": "ok"}
            logger.info("Embedding model: healthy")
        except EmbeddingError as e:
            logger.error(f"Embedding model health check failed: {e}")
            health["status"] = "degraded"
            health["components"]["embedding_model"] = {"status": "error", "reason": str(e)}

        return health
