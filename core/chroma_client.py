import chromadb
from langchain_chroma import Chroma
from core.config import settings
from core.embedding_model import get_embedding_model
from core.logger import get_logger

logger = get_logger(__name__)


def get_vector_store(collection_name: str) -> Chroma:
    logger.info(f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
    raw_client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
    )
    return Chroma(
        client=raw_client,
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
    )
