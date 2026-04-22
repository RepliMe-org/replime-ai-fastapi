# replime-ai-fastapi

The AI backend for the Replime project. Built with FastAPI, it handles YouTube video ingestion, transcript processing, vector storage via ChromaDB, and RAG-based chat using Groq LLMs.

## Architecture Overview

```
routes/          API route handlers (health, ingestion)
services/        Business logic (ingestion pipeline)
rag/             RAG components: chunker, embedder, retriever, LLM client, prompt builder
schemas/         Pydantic request/response models
core/            Config, dependencies, exceptions, logging
```

The ingestion flow: YouTube transcript → chunked → embedded (HuggingFace) → stored in ChromaDB.

## Requirements

- Python 3.11+
- A running **ChromaDB** instance (default: `localhost:8001`)
- A **Groq API key** (for LLM inference)
- A **Hugging Face token** (for embedding model downloads)

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
| `CHROMA_HOST` | ChromaDB host (default: `localhost`) |
| `CHROMA_PORT` | ChromaDB port (default: `8001`) |
| `EMBEDDING_MODEL_ID` | HuggingFace sentence-transformer model ID |
| `HF_TOKEN` | HuggingFace API token (for model access) |
| `INTERNAL_TOKEN` | Shared secret used by the Spring Boot backend to authenticate requests |
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `GROQ_MODEL` | Groq model name (e.g. `llama-3.1-8b-instant`) |
| `SPRING_BOOT_BASE_URL` | Base URL of the Spring Boot backend (e.g. `http://localhost:8080`) |

### 4. Start ChromaDB

ChromaDB must be running before starting this service.Run it directly:

```bash
chroma run --port 8001
```

## Run the Server (in another terminal)

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

## API Endpoints

All endpoints require the `Authorization: Bearer <INTERNAL_TOKEN>` header.

### Health Check

```
GET /health
```

Returns service status and ChromaDB connectivity.

### Video Ingestion

```
POST /internal/videos/{video_id}/index
```

Triggers background ingestion of a YouTube video's transcript into the vector store.

**Request body:**
```json
{
  "chatbot_id": "string",
  "youtube_video_id": "string",
  "video_title": "string"
}
```

**Response (202 Accepted):**
```json
{
  "status": "accepted",
  "video_id": "string"
}
```

### Delete Video

```
DELETE /internal/videos/{video_id}
```

Removes all vector chunks for a video from the store.

**Request body:**
```json
{
  "chatbot_id": "string"
}
```

## Testing

```bash
pytest
```

## Postman Collection

Import `AI.postman_collection.json` into Postman to test all endpoints. Set the `base_url` and `internal_token` collection variables before sending requests.
