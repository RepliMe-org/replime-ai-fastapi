import logging
from contextlib import asynccontextmanager

import aio_pika
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.config import settings
from core.exceptions import AppError
from core.logging import setup_logging
from infrastructure.redis_client import close_redis
from rag.retrieval.embedder import get_embedder
from rag.retrieval.vector_store import get_vector_store
from routes import api_router
from infrastructure.http_client import close_http_client
from workers.dlq_handler import DLQHandler
from workers.video_index_worker import VideoIndexWorker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # Dense embedder — load weights into memory once at startup
    logger.info("Loading embedding model at startup...")
    get_embedder()._load()
    logger.info("Embedding model ready.")

    # Qdrant connection + sparse BM25 model warmup
    try:
        logger.info("Connecting to Qdrant and warming sparse model...")
        get_vector_store()._get_sparse_model()
        get_vector_store().healthcheck()
        logger.info("Vector store ready.")
    except Exception as exc:
        logger.warning("Vector store warmup failed (health will report degraded): %s", exc)

    # RabbitMQ consumer — Spring Boot already declared all queues/exchanges
    rabbitmq_connection = None
    try:
        logger.info("Connecting to RabbitMQ at %s...", settings.RABBITMQ_HOST)
        rabbitmq_connection = await aio_pika.connect_robust(
            host=settings.RABBITMQ_HOST,
            port=5672,
            login=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASS,
        )
        channel = await rabbitmq_connection.channel()
        await channel.set_qos(prefetch_count=1)

        main_queue = await channel.get_queue("replime.video.index")
        dlq = await channel.get_queue("replime.video.index.dlq")

        await main_queue.consume(VideoIndexWorker().process)
        await dlq.consume(DLQHandler().handle)
        logger.info("RabbitMQ consumer started.")
    except Exception as exc:
        logger.warning(
            "RabbitMQ connection failed — consumer not started "
            "(HTTP ingestion endpoint still works): %s",
            exc,
        )

    yield

    if rabbitmq_connection:
        await rabbitmq_connection.close()
    await close_redis()
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
