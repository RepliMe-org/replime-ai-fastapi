from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    """Base model exposing camelCase JSON (Spring Boot boundary) over snake_case fields."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ── Inbound: Spring Boot → FastAPI (POST /ai/analytics/process) ────────────────

class QuestionInput(_CamelModel):
    text: str
    # Whether the answer to this question cited at least one video. Unanswered
    # CONTENT_QUESTIONs are the content-gap signal.
    answered_with_sources: bool = True


class AnalyticsRequest(_CamelModel):
    chatbot_id: str
    # Channel description (owned by Spring Boot) — the "what the channel covers" reference.
    description: str | None = None
    questions: list[QuestionInput] = Field(default_factory=list)


# ── Outbound: FastAPI → Spring Boot → frontend ────────────────────────────────

class QuestionCluster(_CamelModel):
    theme: str
    count: int
    example_questions: list[str] = Field(default_factory=list)


class ContentGapCluster(_CamelModel):
    topic: str
    frequency: int
    sample_questions: list[str] = Field(default_factory=list)


class AnalyticsResponse(_CamelModel):
    most_asked_clusters: list[QuestionCluster] = Field(default_factory=list)
    content_gaps: list[ContentGapCluster] = Field(default_factory=list)
    executive_summary: str = ""
