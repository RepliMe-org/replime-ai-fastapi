import logging

from fastapi import APIRouter, Depends

from core.dependencies import verify_internal_token
from rag.retrieval.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", dependencies=[Depends(verify_internal_token)])
def health_check():
    health = {"status": "ok", "service": "ai-fastapi", "components": {}}

    try:
        get_vector_store().healthcheck()
        health["components"]["qdrant"] = {"status": "ok"}
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        health["status"] = "degraded"
        health["components"]["qdrant"] = {"status": "error", "reason": str(e)}

    return health
