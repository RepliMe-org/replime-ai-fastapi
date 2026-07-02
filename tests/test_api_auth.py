"""API tests for the X-Internal-Token guard (core/dependencies.py)."""
from unittest.mock import MagicMock

import routes.health as health
from core.config import settings


def test_missing_token_rejected(raw_client, monkeypatch):
    monkeypatch.setattr(settings, "X_INTERNAL_TOKEN", "secret")
    assert raw_client.get("/ai/health").status_code == 401


def test_wrong_token_rejected(raw_client, monkeypatch):
    monkeypatch.setattr(settings, "X_INTERNAL_TOKEN", "secret")
    r = raw_client.get("/ai/health", headers={"X-Internal-Token": "wrong"})
    assert r.status_code == 401


def test_valid_token_allows_access(raw_client, monkeypatch):
    monkeypatch.setattr(settings, "X_INTERNAL_TOKEN", "secret")
    vs = MagicMock()
    vs.healthcheck = MagicMock()
    monkeypatch.setattr(health, "get_vector_store", lambda: vs)
    r = raw_client.get("/ai/health", headers={"X-Internal-Token": "secret"})
    assert r.status_code == 200
