import logging

import chromadb
from core.config import settings

logger = logging.getLogger(__name__)


class AIService:
    def health_check(self) -> dict:
        health = {
            "status": "ok",
            "service": "ai-fastapi",
            "components": {}
        }

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
            logger.error("ChromaDB health check failed: %s", e)
            health["status"] = "degraded"
            health["components"]["chroma"] = {"status": "error", "reason": str(e)}

        return health
