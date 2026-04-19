import os
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import settings
from core.exceptions import EmbeddingError
from core.logger import get_logger

logger = get_logger(__name__)

_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _model
    if _model is None:
        try:
            if settings.HF_TOKEN:
                os.environ["HF_TOKEN"] = settings.HF_TOKEN
                logger.info("HF_TOKEN applied from settings")

            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_ID}")
            _model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_ID,
                cache_folder=settings.CACHE_DIR,
            )
            logger.info("Embedding model ready")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Could not load embedding model '{settings.EMBEDDING_MODEL_ID}'") from e
    return _model
