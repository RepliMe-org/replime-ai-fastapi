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


class NonRetryableIngestionError(Exception):
    """Ingestion stage failed permanently — retrying will never help."""
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage}] {reason}")


class RetryableIngestionError(Exception):
    """Ingestion stage failed transiently — retrying may succeed."""
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage}] {reason}")
