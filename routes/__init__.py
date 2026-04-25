from fastapi import APIRouter

from .health import router as health_router
from .ingestion import router as ingestion_router
from .chat import router as chat_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/ai", tags=["health"])
api_router.include_router(ingestion_router, prefix="/ai", tags=["ingestion"])
api_router.include_router(chat_router, prefix="/ai", tags=["chat"])

__all__ = ["api_router"]