import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Application settings
    APP_NAME: str = "Replime AI FastAPI"
    APP_VERSION: str = "0.1.0"

    # Security settings — required for internal endpoints
    INTERNAL_TOKEN: str = ""

    # ChromaDB connection settings
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_PATH: str = ".chroma"

    # Embedding model settings
    EMBEDDING_MODEL_ID: str = "paraphrase-multilingual-MiniLM-L12-v2"
    TRANSFORMERS_OFFLINE: int = 1  # Set to 1 to force offline mode for HuggingFace models
    CACHE_DIR: str = ".cache/models"

    # HuggingFace (optional — set in .env for faster downloads)
    HF_TOKEN: Optional[str] = None

    # Groq LLM settings
    GROQ_API_KEY: str = ""
    GROQ_CHAT_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_REWRITE_MODEL: str = "llama-3.1-8b-instant"
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.4
    

    # Spring Boot backend
    SPRING_BOOT_BASE_URL: str = "http://localhost:8080/api/v1"

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
