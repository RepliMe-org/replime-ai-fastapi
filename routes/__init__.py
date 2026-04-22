from fastapi import APIRouter

from .health import router as health_router
from .ingestion import router as ingestion_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/ai", tags=["health"])
api_router.include_router(ingestion_router, prefix="/ai", tags=["ingestion"])

__all__ = ["api_router"]