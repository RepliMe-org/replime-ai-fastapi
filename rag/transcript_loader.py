import logging

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from core.exceptions import TranscriptError

logger = logging.getLogger(__name__)

# Ordered by preference: standard codes first, then regional variants
_LANGUAGE_PRIORITY = ["en", "ar", "en-GB", "en-US", "en-AU", "en-CA", "en-IN"]


def load_transcript(youtube_video_id: str) -> list[dict]:
    api = YouTubeTranscriptApi()

    # Attempt 1: preferred language list
    try:
        segments = api.fetch(youtube_video_id, languages=_LANGUAGE_PRIORITY)
        logger.info("Loaded transcript via youtube_transcript_api")
        return [{"text": s.text, "start": s.start} for s in segments]
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning("Transcript unavailable via priority languages for %s, trying any available", youtube_video_id)
    except Exception:
        logger.warning("Unexpected error fetching transcript for %s, trying any available", youtube_video_id, exc_info=True)

    # Attempt 2: accept any available transcript language
    try:
        transcript_list = api.list(youtube_video_id)
        for transcript in transcript_list:
            segments = transcript.fetch()
            logger.info(
                "Loaded transcript via fallback language=%s youtube_video_id=%s",
                transcript.language_code,
                youtube_video_id,
            )
            return [{"text": s.text, "start": s.start} for s in segments]
    except Exception:
        logger.warning("Any-language fallback also failed for %s", youtube_video_id, exc_info=True)

    raise TranscriptError(f"Could not fetch transcript for {youtube_video_id}")
