# replime-ai-fastapi

The AI backend for the Replime project. Built with FastAPI, it handles YouTube video ingestion, transcript processing, vector storage via ChromaDB, and RAG-based chat using Groq LLMs.

## Architecture Overview

```
routes/          API route handlers (health, ingestion, chat)
services/        Business logic (ingestion pipeline, chat/RAG pipeline)
rag/             RAG components: chunker, embedder, LLM client, prompt builder, query rewriter
schemas/         Pydantic request/response models
core/            Config, dependencies, exceptions, logging
```

**Ingestion flow:** YouTube transcript → chunked → embedded (sentence-transformers) → stored in ChromaDB → callback sent to Spring Boot.

**Chat flow:** query → language detected → intent classified & query rewritten (concurrent) → embedded → retrieved from ChromaDB → prompt built → LLM generates answer → citations extracted → sources returned. Classification (message class) runs async after the response is sent.

## Requirements

- Python 3.11+
- A **Groq API key** (for LLM inference)
- A **Hugging Face token** (optional, for faster model downloads)

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

Edit `.env` and fill in your values:

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Replime AI FastAPI` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `INTERNAL_TOKEN` | *(required)* | Shared secret for internal endpoint auth |
| `QDRANT_URL` | *(required)* | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | *(required)* | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `replime_chunks` | Qdrant collection name (shared, `chatbot_id`-filtered) |
| `EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-large` | Sentence-transformer model ID |
| `TRANSFORMERS_OFFLINE` | `1` | Set to `0` to allow HuggingFace downloads |
| `CACHE_DIR` | `.cache/models` | Local model cache directory |
| `HF_TOKEN` | *(optional)* | HuggingFace token for faster downloads |
| `GROQ_API_KEY` | *(required)* | Groq API key for LLM inference |
| `GROQ_CHAT_MODEL` | `llama-3.3-70b-versatile` | Groq model for chat answers |
| `GROQ_REWRITE_MODEL` | `llama-3.3-70b-versatile` | Groq model for query rewriting |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.4` | Minimum similarity score to include a chunk |
| `SPRING_BOOT_BASE_URL` | `http://localhost:8080/api/v1` | Spring Boot backend base URL |

### 4. Run the server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

## API Endpoints

All endpoints require the `X-Internal-Token: <INTERNAL_TOKEN>` header.

All paths are prefixed with `/ai` (e.g. `GET /ai/health`).

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

Queues background ingestion of one or more YouTube videos into the vector store. Returns immediately with `202 Accepted`. A callback is sent to Spring Boot per video when ingestion completes.

**Request body:**
```json
{
  "chatbot_id": "ali_muhammad_ali",
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

Removes all vector chunks for a video from the store.

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
    "formality": "NEUTRAL"
  },
  "first_message": true
}
```

`conversation_history` roles must be `"USER"` or `"BOT"`. Pass `message_classes: []` to skip classification. `verbosity` accepts `"CONCISE"`, `"BALANCED"`, or `"DETAILED"`. `tone` and `formality` accept `"NEUTRAL"`, `"FRIENDLY"` / `"CASUAL"`, `"ENCOURAGING"` / `"FORMAL"`, `"HUMOROUS"`.

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
  ]
}
```

`session_title` is only present when `first_message: true`. `sources` lists one entry per unique video cited in the answer. Classification (`message_classes`) runs asynchronously after the response is returned — pass an empty list to skip it.

**Other query scenarios handled by the pipeline:**

| Query type | Example | Behaviour |
|---|---|---|
| Content question | `ما أسباب التسويف وكيف نتغلب عليه؟` | Full RAG pipeline, sources returned |
| Greeting | `مرحبا` | Short-circuit, no retrieval |
| Out-of-scope | `ما هو طقس القاهرة اليوم؟` | Short-circuit, no retrieval |
| Vague / too short | `س` | Asks a clarifying question |

## Testing

```bash
pytest
```

## Postman Collection

Import `AI.postman_collection.json` into Postman to test all endpoints. Set the `api` collection variable to `http://localhost:8000/ai` and add your `X-Internal-Token` header value before sending requests.
