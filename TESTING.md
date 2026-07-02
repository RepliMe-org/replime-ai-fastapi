# Testing & Evaluation

This document describes how the Replime AI FastAPI service is tested.

## Philosophy

Tests target **deterministic unit and API behaviour**. Every external dependency
— Qdrant, the LLM providers, the embedding model, Redis, RabbitMQ, and the HTTP
client to Spring Boot — is **mocked in the default suite**, so tests need no
network, no API keys, and no running services. They are fast (the whole suite
runs in ~2 s) and repeatable.

Tests that need a real model or live service (e.g. the embedding smoke test) are
marked `@pytest.mark.integration` and **deselected by default** (see
`pytest.ini`).

## Running the tests

```bash
# install dev tooling (pytest, pytest-asyncio, pytest-cov)
pip install -r requirements-dev.txt

# default suite (mocked; fast, deterministic)
pytest

# with coverage
pytest --cov=core --cov=rag --cov=services --cov=routes --cov=schemas \
       --cov=infrastructure --cov=workers --cov-report=term-missing

# opt-in integration tests (loads a real sentence-transformers model)
pytest -m integration
```

## Test layers

| Layer | Scope | Files |
|---|---|---|
| Text-processing | Language detection, Arabic normalization, chunking + timestamp mapping | `test_language_detector.py`, `test_text_normalizer.py`, `test_chunker.py` |
| Retrieval | MMR reranking, vector-store query/mapping/dedup, dominant-language ratio | `test_mmr.py`, `test_vector_store.py` |
| Contract | Pydantic request/response schemas, camelCase boundary, config routing | `test_schemas.py`, `test_config.py` |
| LLM / adapter | Model-spec parsing, cross-provider fallback chain, injection guard, prompt assembly, task wrappers | `test_llm_client.py`, `test_intent_classifier.py`, `test_prompt_builder.py`, `test_llm_tasks.py` |
| Service | Chat pipeline, analytics, classification, ingestion stages, corpus-language, description regen | `test_chat_service.py`, `test_citation_extraction.py`, `test_analytics_service.py`, `test_classification_service.py`, `test_ingestion_service.py`, `test_corpus_language_service.py`, `test_description_service.py` |
| API | FastAPI routes via `TestClient` + `app.dependency_overrides` | `test_api_auth.py`, `test_api_chat.py`, `test_api_ingestion.py`, `test_api_health.py`, `test_api_analytics.py` |
| Worker / queue | RabbitMQ consumer idempotency + failure routing, DLQ handler | `test_workers.py` |
| Integration (opt-in) | Real embedding model load | `test_embedder.py` |

## Mocking strategy

| Dependency | Test approach |
|---|---|
| Qdrant client | `MagicMock` injected as `VectorStore._client`; canned points / counts / scroll pages |
| Sparse BM25 model | `MagicMock` injected as `VectorStore._sparse_model` |
| LLM providers (OpenAI-compatible) | `AsyncMock` on `LLMClient.generate`, or `monkeypatch` of `_call_llm` |
| Embedding model | `AsyncMock` on `Embedder.embed_query` / `embed_documents` |
| Redis | `MagicMock` with `AsyncMock` methods; `get_redis` patched |
| HTTP → Spring Boot | `AsyncMock` on the httpx client / callback helpers |
| FastAPI dependencies | `app.dependency_overrides[verify_internal_token]` |
| Background tasks | Executed by `TestClient`; entry points patched to `AsyncMock` |
| aio_pika message | Fake with `message.body` bytes + async-context-manager `message.process()` |

The app **lifespan** (embedder load, Qdrant/RabbitMQ warmup) is skipped in API
tests by constructing `TestClient(app)` without the context-manager form.

## Latest results

- **160 passed, 2 deselected** (integration) in ~2 s.
- **83% line coverage** across `core`, `rag`, `services`, `routes`, `schemas`,
  `infrastructure`, `workers`.

Routes, schemas, prompts, MMR, and text normalization are at or near 100%.
Coverage is intentionally lower on I/O-bound modules that the default suite does
not exercise for real (`transcript_loader` — live YouTube/yt-dlp; `embedder` —
real model download).

## Bug found & fixed via testing

`prompt_builder.build_system_prompt` called `config.verbosity.upper()`
unconditionally, but `verbosity` is `Optional[str]` in the schema. A chat request
that omitted verbosity would raise `AttributeError` and 500. Fixed to
`(config.verbosity or "").upper()` and covered by
`test_prompt_builder.test_verbosity_none_does_not_crash`.

## Limitations

- **Load / performance testing is not done at the AI layer.** End-to-end load
  testing of the chat and analytics endpoints depends on external LLM providers
  reached through **free-tier endpoints**, whose strict rate limits and high,
  variable latency make concurrent-load figures unrepresentative of this
  service's own behaviour. Load and throughput testing is therefore performed at
  the backend layer, which owns the request-rate concerns; the AI layer is
  validated for functional correctness through the tests above.
- **Mocked externals.** The default suite trades full end-to-end validation for
  speed and determinism — a deliberate, standard trade-off.
- **Integration tests are opt-in.** Tests that load the real embedding model are
  deselected by default to keep the suite fast and offline; they are run
  explicitly when validating the model boundary.
