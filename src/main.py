# api.py or main.py (FastAPI 부분)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from graph import build_graph

app = FastAPI()

graph = build_graph()


# =========================
# Request / Response Schema
# =========================

class HistoryTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    question: str
    history: List[HistoryTurn] = []


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    evidence_urls: List[str]


# =========================
# Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    멀티턴 RAG chat endpoint
    """

    # 🔹 LangGraph 초기 state
    state = {
        "question": req.question,
        "history": [
            {"user": h.user, "assistant": h.assistant}
            for h in req.history
        ],
    }

    # 🔹 Graph 실행
    result = graph.invoke(state)

    return {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", ""),
        "evidence_urls": result.get("evidence_urls", []),
    }
