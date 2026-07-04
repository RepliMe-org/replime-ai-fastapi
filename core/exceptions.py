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


class TranscriptRateLimitError(TranscriptError):
    """Raised when YouTube blocks the request due to IP detection or rate-limiting."""
    code = "TRANSCRIPT_IP_BLOCKED"


class VectorStoreError(AppError):
    code = "VECTOR_STORE_ERROR"
    status_code = 503


class InvalidRequestError(AppError):
    code = "INVALID_REQUEST"
    status_code = 422


class LLMError(AppError):
    code = "LLM_ERROR"
    status_code = 503


# EmbeddingError maps to LLMError, so embedding failures surface with code LLM_ERROR.
EmbeddingError = LLMError


# Fallback, per-stage messages shown to end users when a raise site doesn't
# supply a more specific `user_message`. Keep these free of internal exception
# text — `reason` (technical) stays in logs/exceptions only.
_STAGE_USER_MESSAGES = {
    "TRANSCRIPT_EXTRACTION": "We couldn't retrieve a transcript for this video.",
    "CHUNKING": "Something went wrong while processing this video's transcript.",
    "EMBEDDING_GENERATION": "Something went wrong while analyzing this video's content.",
    "VECTOR_INDEXING": "Something went wrong while saving this video's content.",
}
DEFAULT_INGESTION_USER_MESSAGE = "Something went wrong while processing this video. Please try again later."


class NonRetryableIngestionError(Exception):
    """Ingestion stage failed permanently — retrying will never help."""
    def __init__(self, stage: str, reason: str, user_message: str | None = None):
        self.stage = stage
        self.reason = reason
        self.user_message = user_message or _STAGE_USER_MESSAGES.get(stage, DEFAULT_INGESTION_USER_MESSAGE)
        super().__init__(f"[{stage}] {reason}")


class RetryableIngestionError(Exception):
    """Ingestion stage failed transiently — retrying may succeed."""
    def __init__(self, stage: str, reason: str, user_message: str | None = None):
        self.stage = stage
        self.reason = reason
        self.user_message = user_message or _STAGE_USER_MESSAGES.get(stage, DEFAULT_INGESTION_USER_MESSAGE)
        super().__init__(f"[{stage}] {reason}")
