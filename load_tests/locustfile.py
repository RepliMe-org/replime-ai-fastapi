"""Locust load test for the Replime AI FastAPI endpoints.

This measures END-TO-END latency under concurrent load — the parts that dominate
real response time (query embedding + Qdrant hybrid retrieval + LLM generation).
Those can only be measured against a running server with real Qdrant and LLM
credentials, so this is run manually, not in CI.

Prerequisites
-------------
1. The FastAPI server is running and reachable (see CLAUDE.md to start it).
2. At least one chatbot has indexed content in Qdrant (otherwise chat requests
   short-circuit to the "no content" fallback and won't exercise the LLM path).

Setup
-----
    pip install locust                 # already in requirements-dev.txt
    # The internal token is read automatically from the project .env.
    # Just point at a chatbot that actually has indexed content:
    export REPLIME_CHATBOT_ID="<a chatbot_id with indexed videos>"

    # If you export X_INTERNAL_TOKEN by hand instead (e.g. to also use it with
    # curl) on WSL against a Windows-edited (CRLF) .env, strip the trailing \r
    # or every request will fail instantly with a malformed-header error:
    #   export X_INTERNAL_TOKEN=$(grep -E '^X_INTERNAL_TOKEN=' .env | cut -d= -f2- | tr -d '\r')

Run (headless: 20 users, 5/s spawn, 60 s)
-----------------------------------------
    locust -f load_tests/locustfile.py --host http://localhost:8000 \
           --users 20 --spawn-rate 5 --run-time 60s --headless

Or open the interactive web UI at http://localhost:8089
    locust -f load_tests/locustfile.py --host http://localhost:8000
"""
import os
import random

from dotenv import load_dotenv
from locust import HttpUser, between, task

# Auto-load the project .env so the internal token is picked up without exporting
# it by hand. X_INTERNAL_TOKEN is the server's own setting name; INTERNAL_TOKEN is
# accepted as an alias. An explicit shell export still overrides the .env value.
# .strip() guards against a stray trailing \r — a CRLF-terminated .env (e.g. edited
# on Windows) or a naive `grep | cut` extraction leaves one on the token, which
# corrupts the HTTP header framing and makes every single request fail instantly.
load_dotenv()
INTERNAL_TOKEN = (os.getenv("X_INTERNAL_TOKEN") or os.getenv("INTERNAL_TOKEN", "")).strip()
CHATBOT_ID = os.getenv("REPLIME_CHATBOT_ID", "demo-chatbot").strip()

if not INTERNAL_TOKEN:
    print("[locustfile] WARNING: no X_INTERNAL_TOKEN/INTERNAL_TOKEN found — requests will 401.")

_HEADERS = {"X-Internal-Token": INTERNAL_TOKEN}

# A mix of English and Arabic queries so both model-routing paths are exercised.
_QUERIES = [
    "What is the main topic of this channel?",
    "Can you summarize the key points for a beginner?",
    "How does this work in practice?",
    "What tools or resources were recommended?",
    "ما هو الموضوع الرئيسي لهذه القناة؟",
    "لخص لي أهم النقاط التي تم ذكرها",
]

_ANALYTICS_QUESTIONS = [
    {"text": "How do I get started?", "answeredWithSources": True},
    {"text": "What is the pricing?", "answeredWithSources": False},
    {"text": "Which framework is best for beginners?", "answeredWithSources": True},
]


def _chat_payload(query: str, first_message: bool = False) -> dict:
    return {
        "chatbot_id": CHATBOT_ID,
        "message_id": random.randint(1, 1_000_000),
        "query": query,
        "conversation_history": [],
        "message_classes": [],
        "config": {
            "chatbot_name": "LoadTestBot",
            "talk_like_me": False,
            "tone": "FRIENDLY",
            "verbosity": "BALANCED",
            "formality": "NEUTRAL",
        },
        "first_message": first_message,
    }


class ReplimeAIUser(HttpUser):
    # Think time between requests, emulating a user reading a reply before asking again.
    wait_time = between(1, 3)

    @task(1)
    def health(self):
        self.client.get("/ai/health", headers=_HEADERS, name="GET /ai/health")

    @task(6)
    def chat(self):
        # The heaviest path: embedding + hybrid retrieval + LLM generation.
        self.client.post(
            "/ai/chat/process",
            json=_chat_payload(random.choice(_QUERIES)),
            headers=_HEADERS,
            name="POST /ai/chat/process",
        )

    @task(2)
    def analytics(self):
        self.client.post(
            "/ai/analytics/process",
            json={"chatbotId": CHATBOT_ID, "questions": _ANALYTICS_QUESTIONS},
            headers=_HEADERS,
            name="POST /ai/analytics/process",
        )
