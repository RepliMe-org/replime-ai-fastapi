from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    """Base model exposing camelCase JSON (Spring Boot boundary) over snake_case fields."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ── Inbound: Spring Boot → FastAPI (POST /ai/analytics/process) ────────────────

class CitedVideoInput(_CamelModel):
    youtube_video_id: str
    video_title: str
    citation_count: int = 1  # Spring Boot may pre-aggregate; raw events default to 1


class ContentGapInput(_CamelModel):
    query: str
    language: str | None = None


class AnalyticsRequest(_CamelModel):
    chatbot_id: str
    questions: list[str] = Field(default_factory=list)
    content_gaps: list[ContentGapInput] = Field(default_factory=list)
    cited_videos: list[CitedVideoInput] = Field(default_factory=list)


# ── Outbound: FastAPI → Spring Boot → frontend ────────────────────────────────

class QuestionCluster(_CamelModel):
    theme: str
    count: int
    example_questions: list[str] = Field(default_factory=list)


class ContentGapCluster(_CamelModel):
    topic: str
    frequency: int
    sample_questions: list[str] = Field(default_factory=list)


class CitedVideoStat(_CamelModel):
    youtube_video_id: str
    video_title: str
    citation_count: int


class AnalyticsResponse(_CamelModel):
    most_asked_clusters: list[QuestionCluster] = Field(default_factory=list)
    content_gaps: list[ContentGapCluster] = Field(default_factory=list)
    most_cited_videos: list[CitedVideoStat] = Field(default_factory=list)
    executive_summary: str = ""
    content_opportunities: list[str] = Field(default_factory=list)
