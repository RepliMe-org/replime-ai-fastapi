import logging

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from core.exceptions import TranscriptError

logger = logging.getLogger(__name__)




def load_transcript(youtube_video_id: str) -> list[dict]:
    try:
        segments = YouTubeTranscriptApi.get_transcript(
            youtube_video_id, languages=["en", "ar"]
        )
        result = [{"text": s["text"], "start": s["start"]} for s in segments]
        logger.info("Loaded transcript via youtube_transcript_api", extra={"youtube_video_id": youtube_video_id})
        return result
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning("Transcript unavailable via API for %s, trying fallback", youtube_video_id)
    except Exception:
        logger.warning("Unexpected error fetching transcript for %s, trying fallback", youtube_video_id, exc_info=True)

    try:
        from langchain_community.document_loaders import YoutubeLoader

        docs = YoutubeLoader.from_youtube_url(
            f"https://youtube.com/watch?v={youtube_video_id}"
        ).load()
        result = [{"text": doc.page_content, "start": None} for doc in docs]
        logger.info("Loaded transcript via YoutubeLoader fallback", extra={"youtube_video_id": youtube_video_id})
        return result
    except Exception:
        logger.warning("YoutubeLoader fallback also failed for %s", youtube_video_id, exc_info=True)

    raise TranscriptError(f"Could not fetch transcript for {youtube_video_id}")
