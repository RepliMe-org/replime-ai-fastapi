class AppError(Exception):
    message: str
    code: str
    status_code: int = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ServiceAuthError(AppError):
    code = "UNAUTHORIZED"
    status_code = 401


class TranscriptError(AppError):
    code = "TRANSCRIPT_FAILED"
    status_code = 422


class VectorStoreError(AppError):
    code = "VECTOR_STORE_ERROR"
    status_code = 503


class LLMError(AppError):
    code = "LLM_ERROR"
    status_code = 503


# Aliases for backwards compatibility with existing callers
TranscriptNotFoundError = TranscriptError
EmptyTranscriptError = TranscriptError
ChunkingError = TranscriptError
EmbeddingError = LLMError
VectorStoreConnectionError = VectorStoreError
RetrievalError = VectorStoreError
IngestionError = VectorStoreError
