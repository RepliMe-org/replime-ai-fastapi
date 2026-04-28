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

**Chat flow:** query → rewritten → embedded → retrieved from ChromaDB → prompt built → LLM generates answer → sources returned.

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

| Variable | Description |
|---|---|
| `APP_NAME` | Application name (default: `Replime AI FastAPI`) |
| `APP_VERSION` | Application version |
| `CHROMA_PATH` | ChromaDB persistent storage path (default: `.chroma`) |
| `EMBEDDING_MODEL_ID` | Sentence-transformer model ID |
| `HF_TOKEN` | HuggingFace API token (optional) |
| `INTERNAL_TOKEN` | Shared secret used by Spring Boot to authenticate requests |
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `GROQ_MODEL` | Groq model name (e.g. `llama-3.1-8b-instant`) |
| `SPRING_BOOT_BASE_URL` | Base URL of the Spring Boot backend (e.g. `http://localhost:8080`) |

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

Returns service status and ChromaDB connectivity.

**Response:**
```json
{
  "status": "ok",
  "service": "ai-fastapi",
  "components": {
    "chroma": { "status": "ok" }
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
  "chatbot_id": "chatbot-001",
  "videos": [
    {
      "youtube_video_id": "dQw4w9WgXcQ",
      "video_title": "How I grew to 1M subscribers"
    }
  ]
}
```

**Response (202 Accepted):**
```json
{
  "status": "accepted",
  "chatbot_id": "chatbot-001",
  "total": 1
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

**Request body:**
```json
{
  "session_id": "session-001",
  "chatbot_id": "chatbot-001",
  "query": "What did the speaker say about consistency?",
  "language": "en",
  "conversation_history": [],
  "config": {
    "chatbot_name": "Music Bot",
    "persona_description": "An expert on music videos and lyrics",
    "persona_keywords": ["music", "lyrics", "artist"],
    "tone": "informative",
    "response_length": "detailed",
    "top_k": 10,
    "similarity_threshold": 0.3,
    "max_context_turns": 10
  }
}
```

**Response:**
```json
{
  "answer": "The speaker emphasized...",
  "sources": [
    {
      "video_title": "How I grew to 1M subscribers",
      "chunk_text": "...",
      "youtube_url": "https://youtube.com/watch?v=dQw4w9WgXcQ&t=90s",
      "timestamp_seconds": 90,
      "similarity_score": 0.92
    }
  ],
  "retrieval_ms": 45,
  "llm_ms": 820,
  "rewritten_query": "What did the speaker say about consistency?"
}
```

## Testing

```bash
pytest
```

## Postman Collection

Import `AI.postman_collection.json` into Postman to test all endpoints. Set the `api` collection variable to `http://localhost:8000/ai` and add your `X-Internal-Token` header value before sending requests.
