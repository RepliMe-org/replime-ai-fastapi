from fastapi import APIRouter
from .ai_routes import router as ai_router

api_router = APIRouter()
api_router.include_router(ai_router, prefix="/ai", tags=["ai"])

__all__ = ["api_router"]