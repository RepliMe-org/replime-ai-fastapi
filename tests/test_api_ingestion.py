"""API tests for the ingestion routes: index (202), delete, and list videos."""
from unittest.mock import AsyncMock, MagicMock

import routes.ingestion as ing_route


def test_index_videos_accepts_and_returns_202(app_client, monkeypatch):
    monkeypatch.setattr(ing_route, "run_ingestion", AsyncMock())
    payload = {"chatbot_id": "cb", "videos": [{"youtube_video_id": "v1", "video_title": "T"}]}
    r = app_client.post("/ai/ingest/videos", json=payload)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "ACCEPTED"
    assert body["total"] == 1


def test_delete_video_returns_deleted_count(app_client, monkeypatch):
    vs = MagicMock()
    vs.delete_by_video_id = MagicMock(return_value=5)
    monkeypatch.setattr(ing_route, "get_vector_store", lambda: vs)
    monkeypatch.setattr(ing_route, "refresh_corpus_language", AsyncMock())
    monkeypatch.setattr(ing_route, "refresh_channel_description", AsyncMock())
    r = app_client.request("DELETE", "/ai/delete/video", json={"chatbot_id": "cb", "youtube_video_id": "v1"})
    assert r.status_code == 200
    assert r.json()["deleted_chunks"] == 5


def test_delete_video_noop_skips_refresh(app_client, monkeypatch):
    vs = MagicMock()
    vs.delete_by_video_id = MagicMock(return_value=0)
    monkeypatch.setattr(ing_route, "get_vector_store", lambda: vs)
    refresh_lang = AsyncMock()
    monkeypatch.setattr(ing_route, "refresh_corpus_language", refresh_lang)
    monkeypatch.setattr(ing_route, "refresh_channel_description", AsyncMock())
    r = app_client.request("DELETE", "/ai/delete/video", json={"chatbot_id": "cb", "youtube_video_id": "v1"})
    assert r.status_code == 200
    assert r.json()["deleted_chunks"] == 0
    refresh_lang.assert_not_called()


def test_list_videos_aggregates_totals(app_client, monkeypatch):
    vs = MagicMock()
    vs.list_videos = MagicMock(
        return_value={"cb1": [{"youtube_video_id": "v1", "video_title": "A", "chunk_count": 3}]}
    )
    monkeypatch.setattr(ing_route, "get_vector_store", lambda: vs)
    body = app_client.get("/ai/videos").json()
    assert body["total_chatbots"] == 1
    assert body["total_videos"] == 1
    assert body["total_chunks"] == 3
