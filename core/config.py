from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Replime AI FastAPI"
    APP_VERSION: str = "0.1.0"

    # ChromaDB connection settings
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # Embedding model settings
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    CACHE_DIR: str = ".cache/models"


    HF_TOKEN: str = ""
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

