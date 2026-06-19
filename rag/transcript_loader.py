import json
import logging
import os
import tempfile

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked, NoTranscriptFound, TranscriptsDisabled

from core.config import settings
from core.exceptions import TranscriptError, TranscriptRateLimitError

logger = logging.getLogger(__name__)

# Ordered by preference: standard codes first, then regional variants
_LANGUAGE_PRIORITY = ["en", "ar", "en-GB", "en-US", "en-AU", "en-CA", "en-IN"]


def _load_via_ytdlp(youtube_video_id: str) -> list[dict]:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={youtube_video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "json3",
            "subtitleslangs": ["en", "ar", "en-orig"],
            "outtmpl": os.path.join(tmpdir, "%(id)s"),
            "quiet": True,
            "no_warnings": True,
        }
        cookies_file = settings.YTDLP_COOKIES_FILE
        if cookies_file and os.path.isfile(cookies_file):
            ydl_opts["cookiefile"] = cookies_file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for lang in ["en", "ar", "en-orig"]:
            sub_path = os.path.join(tmpdir, f"{youtube_video_id}.{lang}.json3")
            if not os.path.exists(sub_path):
                continue
            with open(sub_path, encoding="utf-8") as f:
                data = json.load(f)
            segments = []
            for event in data.get("events", []):
                text = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
                text = text.strip()
                if text and text != "\n":
                    segments.append({"text": text, "start": event.get("tStartMs", 0) / 1000})
            if segments:
                logger.info("Loaded transcript via yt-dlp lang=%s youtube_video_id=%s", lang, youtube_video_id)
                return segments

    raise TranscriptError(f"yt-dlp found no subtitles for {youtube_video_id}")


def load_transcript(youtube_video_id: str) -> list[dict]:
    api = YouTubeTranscriptApi()
    ip_blocked = False

    # Attempt 1: preferred language list
    try:
        segments = api.fetch(youtube_video_id, languages=_LANGUAGE_PRIORITY)
        logger.info("Loaded transcript via youtube_transcript_api youtube_video_id=%s", youtube_video_id)
        return [{"text": s.text, "start": s.start} for s in segments]
    except IpBlocked:
        logger.warning("IP blocked by YouTube (attempt 1) youtube_video_id=%s", youtube_video_id)
        ip_blocked = True
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning("Transcript unavailable via priority languages for %s, trying any available", youtube_video_id)
    except Exception:
        logger.warning("Unexpected error fetching transcript for %s, trying any available", youtube_video_id, exc_info=True)

    # Attempt 2: accept any available transcript language (skip if already IP blocked)
    if not ip_blocked:
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
        except IpBlocked:
            logger.warning("IP blocked by YouTube (attempt 2) youtube_video_id=%s", youtube_video_id)
            ip_blocked = True
        except Exception:
            logger.warning("Any-language fallback also failed for %s", youtube_video_id, exc_info=True)

    # Attempt 3: langchain YoutubeLoader — handles auto-generated captions and edge cases
    # that the raw API misses. Skipped when IP blocked (shares the same underlying library).
    if not ip_blocked:
        try:
            from langchain_community.document_loaders import YoutubeLoader

            loader = YoutubeLoader(
                video_id=youtube_video_id,
                language=_LANGUAGE_PRIORITY,
                add_video_info=False,
            )
            docs = loader.load()
            if docs:
                logger.info("Loaded transcript via langchain YoutubeLoader for %s", youtube_video_id)
                return [{"text": docs[0].page_content, "start": 0.0}]
        except Exception:
            logger.warning("Langchain YoutubeLoader fallback failed for %s", youtube_video_id, exc_info=True)

    # Attempt 4: yt-dlp — different HTTP fingerprint, bypasses IP detection and handles
    # additional caption formats. Tried when IP blocked OR all previous attempts failed.
    try:
        return _load_via_ytdlp(youtube_video_id)
    except TranscriptError:
        raise
    except Exception:
        logger.warning("yt-dlp fallback failed for %s", youtube_video_id, exc_info=True)

    if ip_blocked:
        raise TranscriptRateLimitError(f"IP blocked by YouTube for {youtube_video_id}")
    raise TranscriptError(f"Could not fetch transcript for {youtube_video_id}")
