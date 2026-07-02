"""Shared pytest fixtures and fakes for the Replime AI test suite.

The default suite mocks every external dependency (Qdrant, LLM providers, the
embedding model, Redis, and the HTTP client to Spring Boot) so tests are fast,
deterministic, and require no network, API keys, or running services.
"""
import sys
from pathlib import Path

import pytest

# Make the project root importable even when pytest is invoked from elsewhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from core.dependencies import verify_internal_token  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture
def app_client():
    """TestClient with the internal-token guard overridden (authorized calls).

    Built without the context-manager form so the app lifespan (embedder load,
    Qdrant/RabbitMQ warmup) never runs during route tests.
    """
    app.dependency_overrides[verify_internal_token] = lambda: None
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def raw_client():
    """TestClient with no dependency overrides — used for auth tests."""
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
