from fastapi import APIRouter
from .health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, prefix="/ai", tags=["health"])

__all__ = ["api_router"]