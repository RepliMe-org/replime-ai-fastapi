"""API tests for GET /ai/health."""
from unittest.mock import MagicMock

import routes.health as health


def test_health_ok(app_client, monkeypatch):
    vs = MagicMock()
    vs.healthcheck = MagicMock()
    monkeypatch.setattr(health, "get_vector_store", lambda: vs)
    body = app_client.get("/ai/health").json()
    assert body["status"] == "ok"
    assert body["components"]["qdrant"]["status"] == "ok"


def test_health_degraded_when_qdrant_unreachable(app_client, monkeypatch):
    vs = MagicMock()
    vs.healthcheck = MagicMock(side_effect=RuntimeError("connection refused"))
    monkeypatch.setattr(health, "get_vector_store", lambda: vs)
    r = app_client.get("/ai/health")
    assert r.status_code == 200  # health endpoint itself still responds
    body = r.json()
    assert body["status"] == "degraded"
    assert body["components"]["qdrant"]["status"] == "error"
