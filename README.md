# replime-ai-fastapi

The AI backend for the Replime project. Built with FastAPI, it processes YouTube videos into a Qdrant vector store (hybrid dense + sparse search), answers user questions through a multi-provider RAG pipeline, maintains each chatbot's channel description and dominant content language, and clusters audience questions into analytics.

## Architecture Overview

```
routes/          API route handlers (health, ingestion, chat, analytics)
workers/          RabbitMQ consumers (video indexing, dead-letter handling)
services/        Business logic (ingestion pipeline, chat/RAG, description regen, corpus language, analytics, classification)
rag/
  text/          Transcript loading, language detection, normalization, chunking
  retrieval/     Embedder, Qdrant vector store, MMR reranking
  llm/           LLM client (multi-provider), prompts, intent classifier, query rewriter, title/description generators
infrastructure/  HTTPX client, Redis client (idempotency, locks, corpus-language cache)
schemas/         Pydantic request/response models
core/            Config, dependencies, exceptions, logging
```

**Ingestion flow** (two entry points, same pipeline):
- **RabbitMQ** (primary) — Spring Boot publishes to `replime.video.index`; `VideoIndexWorker` consumes it, checks a Redis idempotency key, runs the pipeline, and PATCHes the result back. Unhandled errors NACK without requeue, routing the message to `replime.video.index.dlq`, handled by `DLQHandler`.
- **HTTP** (`POST /ai/ingest/videos`) — same pipeline, run as a FastAPI background task, single attempt.

Pipeline: transcript → language detected → normalized/chunked → embedded (`intfloat/multilingual-e5-large`) → indexed in Qdrant (dense + BM25 sparse) → corpus language refreshed → channel description regenerated in the background (best-effort, serialized per chatbot via a Redis lock) → status callback to Spring Boot.

**Chat flow** (`POST /ai/chat/process`): query → language detected → intent classified & query rewritten (concurrent) → short-circuit if intent ≠ `CONTENT_QUESTION` → embedded → hybrid retrieval from Qdrant (dense cosine + BM25, RRF fusion) → optional MMR diversity reranking → prompt built → LLM answers (model chosen per task, with an Arabic-corpus override and a cross-provider fallback chain) → citations extracted → sources deduplicated. Message classification (`message_classes`) runs in the background after the response is sent.

**Analytics** (`POST /ai/analytics/process`): given a chatbot's channel description and a list of asked questions, an LLM clusters the most-asked topics, flags content gaps (unanswered questions), and writes an executive summary.

## Requirements

- Python 3.11+
- At least one LLM provider API key (Groq, Cerebras, Gemini, or NVIDIA NIM)
- A Qdrant Cloud cluster
- RabbitMQ and Redis (RabbitMQ consumer and idempotency/locking are best-effort — the app still runs without them, but async ingestion and lock-based description regen won't work)
- A Hugging Face token (optional, for faster model downloads)

## Setup

### 1. Create a Python environment

Using Conda (recommended):

```bash
conda create -n replime python=3.11 -y
conda activate replime
```

Or using venv:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values (see `.env.example` for the full commented list):

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Replime AI FastAPI` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `X_INTERNAL_TOKEN` | *(required)* | Shared secret for internal endpoint auth |
| `QDRANT_URL` | *(required)* | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | *(required)* | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `replime_chunks` | Qdrant collection name (shared, `chatbot_id`-filtered) |
| `SPARSE_MODEL_ID` | `Qdrant/bm25` | Sparse (BM25) model ID |
| `EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-large` | Sentence-transformer model ID |
| `TRANSFORMERS_OFFLINE` | `1` | Set to `0` to allow HuggingFace downloads |
| `CACHE_DIR` | `.cache/models` | Local model cache directory |
| `HF_TOKEN` | *(optional)* | HuggingFace token for faster downloads |
| `GROQ_API_KEY` / `CEREBRAS_API_KEY` / `GEMINI_API_KEY` / `NVIDIA_API_KEY` | *(as needed)* | LLM provider keys — only the providers referenced by your model specs need a key |
| `CHAT_MODEL` | `nvidia/qwen/qwen3-235b-a22b` | Answer generation (`provider/model`) |
| `REWRITE_MODEL` | `groq/llama-3.3-70b-versatile` | Query rewriting |
| `INTENT_MODEL` | `groq/llama-3.3-70b-versatile` | Intent classification |
| `TITLE_MODEL` | `groq/llama-3.3-70b-versatile` | Session title generation |
| `CLASSIFICATION_MODEL` | `groq/llama-3.3-70b-versatile` | Message classification |
| `DESCRIPTION_MODEL` | `groq/llama-3.3-70b-versatile` | Channel description regeneration |
| `ANALYTICS_MODEL` | `nvidia/openai/gpt-oss-120b` | Question clustering / content gaps |
| `CHAT_MODEL_AR` / `REWRITE_MODEL_AR` / `DESCRIPTION_MODEL_AR` / `ANALYTICS_MODEL_AR` | *(empty)* | Per-task overrides used only when a chatbot's indexed content is Arabic-dominant |
| `CORPUS_ARABIC_RATIO_THRESHOLD` | `0.1` | Share of a chatbot's chunks that must be Arabic to route the `*_AR` model |
| `LLM_FALLBACK_MODELS` | *(empty)* | Comma-separated `provider/model` specs tried in order when a task's primary model errors |
| `DESCRIPTION_SAMPLE_PER_VIDEO` | `6` | Chunks sampled per video for description regeneration |
| `DESCRIPTION_SAMPLE_MAX_CHARS` | `12000` | Max total excerpt chars fed to the description regenerator |
| `DESCRIPTION_LOCK_TTL_SECONDS` | `60` | Per-chatbot description lock lease (auto-extended while a regen runs) |
| `DESCRIPTION_LOCK_WAIT_SECONDS` | `300` | How long a queued regen waits to acquire the lock before skipping |
| `ANALYTICS_MAX_QUESTIONS` | `300` | Max questions sent to the analytics clusterer |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.4` | Minimum dense-prefetch similarity score |
| `RETRIEVAL_PREFETCH_LIMIT` | `20` | Candidates per hybrid branch before RRF fusion |
| `USE_MMR` | `true` | Enable diversity-aware reranking of retrieved chunks |
| `MMR_LAMBDA` | `0.6` | MMR relevance/diversity trade-off (1.0 = pure relevance) |
| `MMR_CANDIDATE_K` | `20` | Candidate pool size fetched before MMR selects `TOP_K` |
| `RABBITMQ_HOST` / `RABBITMQ_USER` / `RABBITMQ_PASS` | `localhost` / `admin` / `admin` | RabbitMQ connection (queues are declared by Spring Boot) |
| `MAX_RETRIES` | `3` | Ingestion attempts before a retryable error becomes permanent |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | Redis connection (idempotency, locks, corpus-language cache) |
| `IDEMPOTENCY_TTL_SECONDS` | `604800` | How long a processed idempotency key is remembered (7 days) |
| `SPRING_BOOT_BASE_URL` | `http://localhost:8080/api/v1` | Spring Boot backend base URL (webhook callbacks) |
| `YOUTUBE_PROXY` | *(optional)* | Proxy for YouTube transcript fetching (`http://user:pass@host:port`) |

### 4. Run the server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

At startup the app loads the embedding model, warms up Qdrant and the sparse BM25 model, and connects a RabbitMQ consumer for `replime.video.index` / `replime.video.index.dlq`. RabbitMQ connection failure is non-fatal — the HTTP ingestion endpoint still works.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                 # default suite (integration tests deselected); mocked externals, fast and offline
pytest -m integration  # opt-in tests that load a real embedding model
```

## API Endpoints

All endpoints require the `X-Internal-Token: <X_INTERNAL_TOKEN>` header, and all paths are prefixed with `/ai`.

### Health Check

```
GET /ai/health
```

Returns service status and Qdrant connectivity.

**Response:**
```json
{
  "status": "ok",
  "service": "ai-fastapi",
  "components": {
    "qdrant": { "status": "ok" }
  }
}
```

### Ingest Videos

```
POST /ai/ingest/videos
```

Queues background ingestion of one or more YouTube videos into the vector store. Returns immediately with `202 Accepted`. A callback is sent to Spring Boot per video when ingestion completes. (This is the HTTP fallback path — Spring Boot normally publishes to RabbitMQ instead, which runs the same pipeline with retries and a dead-letter queue.)

**Request body:**
```json
{
  "chatbot_id": "ali_muhammad_ali",
  "description": "قناة تتحدث عن تطوير الذات وكتب مثل The Slight Edge",
  "videos": [
    { "youtube_video_id": "biCRsdst958", "video_title": "علي وكتاب - الفارق البسيط The Slight Edge" },
    { "youtube_video_id": "22CW76-SARA", "video_title": "علي وكتاب - حل لغز التسويف Solving The Procrastination Puzzle" },
    { "youtube_video_id": "1TlXEW1qp38", "video_title": "علي وكتاب - معادلة التسويف The Procrastination Equation" },
    { "youtube_video_id": "THnrunRXYss", "video_title": "علي وكتاب - قوة الإرادة" }
  ]
}
```

**Response (202 Accepted):**
```json
{
  "status": "ACCEPTED",
  "chatbot_id": "ali_muhammad_ali",
  "total": 4
}
```

### Delete Video

```
DELETE /ai/delete/video
```

Removes all vector chunks for a video from the store. If any chunks were deleted, the chatbot's corpus language and channel description are refreshed in the background from what remains (description becomes `null` if nothing remains).

**Request body:**
```json
{
  "chatbot_id": "chatbot-001",
  "youtube_video_id": "dQw4w9WgXcQ"
}
```

**Response:**
```json
{
  "youtube_video_id": "dQw4w9WgXcQ",
  "deleted_chunks": 6
}
```

### List Videos

```
GET /ai/videos
```

Returns every indexed video grouped by chatbot, with chunk counts — useful for debugging the vector store's current state.

**Response:**
```json
{
  "chatbots": {
    "ali_muhammad_ali": [
      { "youtube_video_id": "biCRsdst958", "video_title": "علي وكتاب - الفارق البسيط The Slight Edge", "chunk_count": 42 }
    ]
  },
  "total_chatbots": 1,
  "total_videos": 1,
  "total_chunks": 42
}
```

### Chat

```
POST /ai/chat/process
```

Runs the full RAG pipeline for a user query and returns an answer with sources.

**Request body — content question (first message):**
```json
{
  "chatbot_id": "ali_muhammad_ali",
  "message_id": 1,
  "query": "ما هو الفارق البسيط وكيف يؤثر على حياتنا؟",
  "conversation_history": [],
  "message_classes": [
    { "id": 0, "name": "النوم" },
    { "id": 1, "name": "التفوق" },
    { "id": 2, "name": "الفلوس" },
    { "id": 3, "name": "الروتين" },
    { "id": 4, "name": "other" }
  ],
  "config": {
    "chatbot_name": "علي وكتاب",
    "talk_like_me": false,
    "verbosity": "BALANCED",
    "tone": "NEUTRAL",
    "formality": "NEUTRAL",
    "description": "قناة تتحدث عن تطوير الذات",
    "topics": ["التسويف", "العادات"]
  },
  "first_message": true
}
```

`conversation_history` roles must be `"USER"` or `"BOT"`. Pass `message_classes: []` to skip classification. `verbosity` accepts `"CONCISE"`, `"BALANCED"`, or `"DETAILED"`. `tone` and `formality` accept `"NEUTRAL"`, `"FRIENDLY"` / `"CASUAL"`, `"ENCOURAGING"` / `"FORMAL"`, `"HUMOROUS"`. `config.description` and `config.topics` are optional seeds used to bootstrap domain-aware intent before any video has been ingested.

**Response:**
```json
{
  "answer": "الفارق البسيط هو مفهوم يقول إن الأفعال الصغيرة المتكررة...",
  "session_title": "مفهوم الفارق البسيط وتأثيره على الحياة",
  "sources": [
    {
      "video_id": "biCRsdst958",
      "video_title": "علي وكتاب - الفارق البسيط The Slight Edge",
      "youtube_url": "https://youtube.com/watch?v=biCRsdst958&t=312s"
    }
  ],
  "intent": "CONTENT_QUESTION",
  "message_id": 1
}
```

`session_title` is only present when `first_message: true`. `sources` lists one entry per unique video cited in the answer. `intent` is one of `GREETING`, `SMALL_TALK`, `CONTENT_QUESTION`, `OUT_OF_SCOPE`, `HARMFUL`. Classification (`message_classes`) runs asynchronously after the response is returned — pass an empty list to skip it.

**Other query scenarios handled by the pipeline:**

| Query type | Example | Behaviour |
|---|---|---|
| Content question | `ما أسباب التسويف وكيف نتغلب عليه؟` | Full RAG pipeline, sources returned |
| Greeting | `مرحبا` | Short-circuit, no retrieval |
| Out-of-scope | `ما هو طقس القاهرة اليوم؟` | Short-circuit, no retrieval |
| Vague / too short | `س` | Asks a clarifying question |

### Analytics

```
POST /ai/analytics/process
```

Clusters a chatbot's audience questions into "most asked" topics and content gaps, and produces an executive summary. Intended to be called periodically by Spring Boot with the questions asked since the last run.

**Request body:**
```json
{
  "chatbotId": "ali_muhammad_ali",
  "description": "قناة تتحدث عن تطوير الذات وكتب مثل The Slight Edge",
  "questions": [
    { "text": "ما هو الفارق البسيط؟", "answeredWithSources": true },
    { "text": "ما رأيك في الطقس؟", "answeredWithSources": false }
  ]
}
```

**Response:**
```json
{
  "mostAskedClusters": [
    { "theme": "الفارق البسيط", "count": 12, "exampleQuestions": ["ما هو الفارق البسيط؟"] }
  ],
  "contentGaps": [
    { "topic": "أسئلة خارج نطاق المحتوى", "frequency": 3, "sampleQuestions": ["ما رأيك في الطقس؟"] }
  ],
  "executiveSummary": "معظم الأسئلة تدور حول مفهوم الفارق البسيط..."
}
```

Returns empty clusters and an empty summary when `questions` is empty (no LLM call made). This endpoint uses camelCase field names (Spring Boot boundary), unlike the other endpoints.

## Postman Collection

Import `AI.postman_collection.json` into Postman to test all endpoints. Set the `api` collection variable to `http://localhost:8000/ai` and add your `X-Internal-Token` header value before sending requests.
