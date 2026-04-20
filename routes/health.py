import logging

import chromadb
from fastapi import APIRouter, Depends

from core.config import settings
from core.dependencies import verify_internal_token

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", dependencies=[Depends(verify_internal_token)])
def health_check():
    health = {"status": "ok", "service": "ai-fastapi", "components": {}}

    try:
        client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        client.heartbeat()
        health["components"]["chroma"] = {"status": "ok"}
    except Exception as e:
        logger.error("ChromaDB health check failed: %s", e)
        health["status"] = "degraded"
        health["components"]["chroma"] = {"status": "error", "reason": str(e)}

    return health
