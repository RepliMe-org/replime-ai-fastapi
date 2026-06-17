import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAI-compatible base URLs per provider. The model spec "provider/model"
# selects one of these; the matching API key comes from settings.
_LLM_PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Application settings
    APP_NAME: str = "Replime AI FastAPI"
    APP_VERSION: str = "0.1.0"

    # Security settings — required for internal endpoints
    INTERNAL_TOKEN: str = ""

    # Qdrant vector store settings
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "replime_chunks"
    CHATBOT_META_COLLECTION: str = "chatbot_meta"
    SPARSE_MODEL_ID: str = "Qdrant/bm25"

    # Embedding model settings
    EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-large"
    TRANSFORMERS_OFFLINE: int = 1  # Set to 1 to force offline mode for HuggingFace models
    CACHE_DIR: str = ".cache/models"
    EMBEDDING_DOC_PREFIX: str = "passage: "
    EMBEDDING_QUERY_PREFIX: str = "query: "

    # HuggingFace (optional — set in .env for faster downloads)
    HF_TOKEN: Optional[str] = None

    # LLM providers (OpenAI-compatible endpoints)
    GROQ_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Per-task model selection — value format: "provider/model"
    CHAT_MODEL: str = "cerebras/llama-3.3-70b"          # main Q&A responses
    REWRITE_MODEL: str = "groq/llama-3.1-8b-instant"    # query rewriting (needs Arabic context)
    INTENT_MODEL: str = "groq/llama-3.1-8b-instant"     # intent classification
    TITLE_MODEL: str = "groq/llama-3.1-8b-instant"      # session title generation
    CLASSIFICATION_MODEL: str = "groq/llama-3.1-8b-instant"  # message classification
    PROFILE_MODEL: str = "groq/llama-3.1-8b-instant"    # channel profile summarization
    ANALYTICS_MODEL: str = "cerebras/llama-3.3-70b"     # batch analytics clustering/summary

    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.4

    # RabbitMQ (ingestion consumer)
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASS: str = "guest"

    # Redis (idempotency store)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Maximum ingestion attempts before a retryable error becomes permanent
    MAX_RETRIES: int = 3

    def provider_credentials(self, provider: str) -> tuple[str, str]:
        """Resolve an LLM provider name to its (base_url, api_key)."""
        base_url = _LLM_PROVIDER_BASE_URLS.get(provider)
        if base_url is None:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
        api_key = {
            "groq": self.GROQ_API_KEY,
            "cerebras": self.CEREBRAS_API_KEY,
            "gemini": self.GEMINI_API_KEY,
        }[provider]
        return base_url, api_key
    

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
