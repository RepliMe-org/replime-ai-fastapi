import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAI-compatible base URLs per provider. The model spec "provider/model"
# selects one of these; the matching API key comes from settings.
_LLM_PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Application settings
    APP_NAME: str = "Replime AI FastAPI"
    APP_VERSION: str = "0.1.0"

    # Security settings — required for internal endpoints
    X_INTERNAL_TOKEN: str = ""

    # Qdrant vector store settings
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "replime_chunks"
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
    NVIDIA_API_KEY: str = ""

    # Per-task model selection — value format: "provider/model".
    # NVIDIA model ids contain their own "/", so nvidia specs are double-prefixed:
    # "nvidia/<owner>/<model>" → provider=nvidia, model="<owner>/<model>".
    CHAT_MODEL: str = "nvidia/qwen/qwen3-235b-a22b"     # main Q&A responses (Arabic+English, RAG)
    REWRITE_MODEL: str = "groq/llama-3.3-70b-versatile"    # query rewriting (needs Arabic context)
    INTENT_MODEL: str = "groq/llama-3.3-70b-versatile"     # intent classification
    TITLE_MODEL: str = "groq/llama-3.3-70b-versatile"      # session title generation
    CLASSIFICATION_MODEL: str = "groq/llama-3.3-70b-versatile"  # message classification
    DESCRIPTION_MODEL: str = "groq/llama-3.3-70b-versatile"  # channel description summarization
    ANALYTICS_MODEL: str = "nvidia/openai/gpt-oss-120b"  # batch analytics clustering/summary

    # Optional per-task overrides used when a chatbot's indexed content is
    # Arabic-dominant (see services/corpus_language_service.py). Empty = no
    # override, always use the base *_MODEL above — fully backward compatible.
    CHAT_MODEL_AR: str = ""
    REWRITE_MODEL_AR: str = ""
    DESCRIPTION_MODEL_AR: str = ""
    ANALYTICS_MODEL_AR: str = ""

    # Global fallback chain — when a task's primary model errors (rate limit,
    # provider outage, bad response), these "provider/model" specs are tried in
    # order before the call is allowed to fail. Comma-separated; empty disables
    # fallback. Prefer instruct models spanning providers so a single provider
    # outage stays survivable (the client does not strip <think> from output).
    LLM_FALLBACK_MODELS: str = ""

    # Share of a chatbot's indexed chunks that must be Arabic for its corpus to
    # be considered Arabic-dominant (routes CHAT_MODEL_AR/REWRITE_MODEL_AR instead
    # of the base model). Kept low/conservative: even a modest amount of Arabic
    # content means a query in any language can retrieve Arabic chunks.
    CORPUS_ARABIC_RATIO_THRESHOLD: float = 0.1

    # Per-task tuning
    # Channel-description regeneration: sample drawn across ALL the chatbot's videos
    # (evenly spaced within each) so the description reflects current Qdrant state.
    DESCRIPTION_SAMPLE_PER_VIDEO: int = 6       # chunks sampled per video for description regen
    DESCRIPTION_SAMPLE_MAX_CHARS: int = 12000   # max total excerpt chars fed to the regenerator
    DESCRIPTION_LOCK_TTL_SECONDS: int = 60      # per-chatbot lock TTL serializing description regen
    ANALYTICS_MAX_QUESTIONS: int = 300          # max questions sent to the analytics clusterer

    # Retrieval
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.4
    RETRIEVAL_PREFETCH_LIMIT: int = 20          # candidates per hybrid branch before RRF fusion

    # MMR (Maximal Marginal Relevance) — diversity-aware reranking of retrieved chunks
    USE_MMR: bool = True
    MMR_LAMBDA: float = 0.6          # 1.0 = pure relevance, 0.0 = pure diversity
    MMR_CANDIDATE_K: int = 20        # candidate pool size fetched before MMR selects top_k

    # RabbitMQ (ingestion consumer)
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_USER: str = "admin"
    RABBITMQ_PASS: str = "admin"

    # Redis (idempotency store)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    IDEMPOTENCY_TTL_SECONDS: int = 60 * 60 * 24 * 7   # how long a processed idempotencyKey is remembered (7 days)

    # Maximum ingestion attempts before a retryable error becomes permanent
    MAX_RETRIES: int = 3

    # Proxy for YouTube transcript fetching (e.g. "http://user:pass@host:port")
    YOUTUBE_PROXY: Optional[str] = None
    
    def provider_credentials(self, provider: str) -> tuple[str, str]:
        """Resolve an LLM provider name to its (base_url, api_key)."""
        base_url = _LLM_PROVIDER_BASE_URLS.get(provider)
        if base_url is None:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
        api_key = {
            "groq": self.GROQ_API_KEY,
            "cerebras": self.CEREBRAS_API_KEY,
            "gemini": self.GEMINI_API_KEY,
            "nvidia": self.NVIDIA_API_KEY,
        }[provider]
        return base_url, api_key

    def fallback_model_specs(self) -> list[str]:
        """Parsed LLM_FALLBACK_MODELS — trimmed, order preserved, blanks dropped."""
        return [s.strip() for s in self.LLM_FALLBACK_MODELS.split(",") if s.strip()]

    @staticmethod
    def model_for(base_spec: str, ar_override: str, corpus_language: str) -> str:
        """Pick a task's model spec by the chatbot's indexed *corpus* language
        (not the query's language — a query in one language can still retrieve
        chunks in another). Falls back to ``base_spec`` when no override is set."""
        if corpus_language == "ar" and ar_override:
            return ar_override
        return base_spec


    # Spring Boot backend
    SPRING_BOOT_BASE_URL: str = "http://localhost:8080/api/v1"

    def validate_internal_token(self) -> None:
        """Warn if X_INTERNAL_TOKEN is not set in production."""
        if not self.X_INTERNAL_TOKEN:
            import sys
            if os.getenv("ENVIRONMENT") == "production":
                raise ValueError("X_INTERNAL_TOKEN must be set in production environment")
            else:
                print("[WARNING] X_INTERNAL_TOKEN not set — internal endpoints will be protected by empty token", file=sys.stderr)


settings = Settings()
settings.validate_internal_token()
