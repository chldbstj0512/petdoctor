# 답변 생성, LLM 연결
from typing import List, Dict
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY


def generate_answer(
    question: str,
    citations: list,
    history: List[Dict[str, str]] = None,  # 🔥 추가
) -> str:
    """
    Generate an answer grounded only on retrieved evidence.
    Supports multi-turn conversation via history.
    """

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,   # 의료 도메인 → 낮게
        openai_api_key=OPENAI_API_KEY
    )

    # =========================
    # 1️⃣ 이전 대화 요약 (최근 2턴)
    # =========================
    history_text = ""
    if history:
        recent = history[-2:]
        history_text = "\n".join(
            f"보호자: {h['user']}\nAI: {h['assistant']}"
            for h in recent
        )

    # =========================
    # 2️⃣ 근거 텍스트 구성
    # =========================
    evidence_text = "\n".join(
        [f"[{c['id']}] {c['content']}" for c in citations]
    )

    # =========================
    # 3️⃣ Prompt
    # =========================
    prompt = f"""
너는 보호자에게 조언하는 따뜻하고 신중한 AI 반려동물 상담자다.
의학적 진단을 내리지 않으며, 현재 상황에서 병원 내원이 필요한지
아니면 경과 관찰이 가능한지를 설명한다.

이전 대화 맥락:
{history_text if history_text else "없음"}

이전 답변이 있는 경우에는, 그 내용을 이미 보호자가 읽었다고 가정하고,
현재 질문은 그에 대한 후속 질문이므로 앞선 답변을 반복하지 말고 자연스럽게 이어서 설명하라.

아래 제공된 근거(evidence)는 비슷한 사례를 보여준 것이다.
참고 용도로만 활용하도록 하고,
보호자가 이해하기 쉬운 말로 조언하라.

규칙:
- 근거를 그대로 평가하거나 "근거가 부족하다"는 표현을 사용하지 말 것
- 근거가 간접적인 경우에도 증상과 연결하여 설명할 것
- 진단을 단정하지 말 것
- 응급 상황이 의심되면 병원 방문을 권장할 것
- 보호자를 안심시키되, 위험 신호는 분명히 알려줄 것

현재 질문:
{question}

근거:
{evidence_text}

답변:
"""

    response = llm.invoke(prompt)
    return response.content.strip()
