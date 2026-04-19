import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Replime AI FastAPI"
    APP_VERSION: str = "0.1.0"

    # Security settings — required for internal endpoints
    INTERNAL_TOKEN: str = ""

    # ChromaDB connection settings
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # Embedding model settings
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    CACHE_DIR: str = ".cache/models"

    # HuggingFace (optional — set in .env for faster downloads)
    HF_TOKEN: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

    def validate_internal_token(self) -> None:
        """Warn if INTERNAL_TOKEN is not set in production."""
        if not self.INTERNAL_TOKEN:
            import sys
            if os.getenv("ENVIRONMENT") == "production":
                raise ValueError("INTERNAL_TOKEN must be set in production environment")
            else:
                print("[WARNING] INTERNAL_TOKEN not set — internal endpoints will be protected by empty token", file=sys.stderr)


settings = Settings()
settings.validate_internal_token()
