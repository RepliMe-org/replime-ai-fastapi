from .base import router as base_router

__all__ = [
    "base_router",
]

from fastapi import APIRouter
api_router = APIRouter(prefix="/api/v1", tags=["api"])

api_router.include_router(base_router, prefix="/base", tags=["base"])