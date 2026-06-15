import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.exceptions import AppError
from core.logging import setup_logging
from rag.embedder import get_embedder
from rag.vector_store import get_vector_store
from routes import api_router
from services.http_client import close_http_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Loading embedding model at startup...")
    get_embedder()._load()
    logger.info("Embedding model ready.")
    try:
        logger.info("Connecting to Qdrant and warming sparse model...")
        get_vector_store()._get_sparse_model()
        get_vector_store().healthcheck()
        logger.info("Vector store ready.")
    except Exception as exc:
        logger.warning("Vector store warmup failed (health will report degraded): %s", exc)
    yield
    await close_http_client()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )


app.include_router(api_router)
