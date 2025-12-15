from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    confidence: str          # "상" | "중" | "하"
    evidence_urls: List[str]

    # 내부 분석용 (선택)
    medical_score: Optional[int] = None
    evidence_score: Optional[int] = None
