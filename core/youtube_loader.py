from langchain_community.document_loaders import YoutubeLoader
from core.exceptions import TranscriptNotFoundError, EmptyTranscriptError
from core.logger import get_logger

logger = get_logger(__name__)

def load_transcript(youtube_url: str) -> str:
    logger.info(f"Loading transcript for: {youtube_url}")
    
    try:
        loader = YoutubeLoader.from_youtube_url(
            youtube_url,
            add_video_info=False,
            language=["ar", "en"],
        )
        docs = loader.load()

    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        raise TranscriptNotFoundError(f"Could not load transcript for {youtube_url}") from e

    if not docs:
        raise TranscriptNotFoundError(f"No transcript found for: {youtube_url}")

    full_transcript = " ".join(doc.page_content for doc in docs)

    if not full_transcript.strip():
        raise EmptyTranscriptError("Transcript loaded but was empty.")

    logger.info(f"Transcript loaded: {len(full_transcript)} characters")
    return full_transcript