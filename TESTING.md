# Testing & Evaluation

This document describes how the Replime AI FastAPI service is tested and how its
performance is evaluated.

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
# install dev tooling (pytest, pytest-asyncio, pytest-cov, locust)
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

## Performance evaluation

### Component micro-benchmarks (local, CPU-bound)

Measured with `benchmarks/component_benchmarks.py` (no network, model, or LLM —
reproducible and free to run). Numbers below were measured locally on Python
3.11; they vary with hardware.

```bash
python benchmarks/component_benchmarks.py
```

| Component | Mean | p95 | Throughput |
|---|---|---|---|
| `language_detect` (English, langdetect) | 3.12 ms | 4.01 ms | ~320 ops/s |
| `language_detect` (Arabic fast-path) | 0.020 ms | 0.025 ms | ~49,500 ops/s |
| `normalize_arabic` | 0.011 ms | 0.012 ms | ~89,000 ops/s |
| `chunk_transcript` (60 segments) | 0.63 ms | 0.72 ms | ~1,580 ops/s |
| `mmr_select` (20 → 5, dim 1024) | 0.70 ms | 0.87 ms | ~1,420 ops/s |
| `extract_citations` | 0.012 ms | 0.012 ms | ~84,000 ops/s |
| `build_messages` (5 chunks) | 0.016 ms | 0.016 ms | ~64,000 ops/s |

**Takeaway:** the CPU-bound glue is negligible (sub-millisecond) except English
`langdetect` (~3 ms) — ~150× slower than the Arabic ratio fast-path. End-to-end
chat latency is therefore dominated by the embedding lookup, the Qdrant hybrid
query, and above all the LLM generation call — none of which are CPU-bound here.

### End-to-end load testing (Locust)

End-to-end latency (embedding + Qdrant retrieval + LLM generation) can only be
measured against a running server with real Qdrant and LLM credentials, so it is
run manually. Harness: `load_tests/locustfile.py`.

```bash
# 1. Start the server in one terminal (needs a real .env: Qdrant + LLM keys + X_INTERNAL_TOKEN)
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 2. In a second terminal, find a chatbot_id that has indexed content.
#    `tr -d '\r'` matters on WSL/Linux if .env has Windows (CRLF) line endings —
#    otherwise a trailing \r rides along on the token and corrupts every header,
#    making requests fail instantly with "Invalid HTTP request received".
export X_INTERNAL_TOKEN=$(grep -E '^X_INTERNAL_TOKEN=' .env | cut -d= -f2- | tr -d '\r')
curl -s -H "X-Internal-Token: $X_INTERNAL_TOKEN" http://localhost:8000/ai/videos | python -m json.tool
export REPLIME_CHATBOT_ID="<one whose videos have chunk_count > 0>"

# 3. Run Locust (the token is auto-read from .env; locust is in requirements-dev.txt)
locust -f load_tests/locustfile.py --host http://localhost:8000 \
       --users 20 --spawn-rate 5 --run-time 60s --headless
```

The harness weights requests 6:2:1 across `POST /ai/chat/process`,
`POST /ai/analytics/process`, and `GET /ai/health`, mixing English and Arabic
queries. Record the per-endpoint median / average / p95 latency, RPS, and failure
rate from the Locust summary.

## Limitations

- **Mocked externals.** The default suite trades full end-to-end validation for
  speed and determinism — a deliberate, standard trade-off.
- **LLM-dominated latency.** Chat response time is bounded by the external LLM
  provider, not by this service's own code.
- **Load ceiling.** Concurrency in load testing is limited by the LLM provider's
  rate limits and by running the server + Locust on one machine.
