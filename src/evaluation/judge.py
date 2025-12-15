# LLM as Judge (전문성과 근거)
import json
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY


def judge_answer(question: str, answer: str, citations: list) -> dict:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    evidence_text = "\n".join(
        [f"[{c['id']}] {c['content']}" for c in citations]
    )

    prompt = f"""
당신은 수의학적 지식을 가진 신중하고 친절한 평가자입니다.

아래 질문, 답변, 근거를 바탕으로
챗봇의 답변이 보호자에게 안전하고 적절한 조언을 했는지 평가하세요.

평가는 반드시 제공된 근거(evidence)를 벗어나지 않아야 합니다.
외부 지식이나 개인적 판단을 추가하지 마세요.

---

[질문]
{question}

[답변]
{answer}

[근거]
{evidence_text}

---

[평가 기준] (각 1~5점)

1. 의료적 적절성 점수 (medical_score)
- 증상에 대한 해석이 과도하지 않고 신중한가
- 진단을 단정하지 않았는가
- 응급 상황과 경과 관찰 판단이 안전한가

2. 근거 활용 점수 (evidence_score)
- 근거의 범위를 벗어나지 않았는가
- 근거와 질문 증상을 무리 없이 연결했는가
- 종(강아지/고양이)에 맞는 조언을 했는가

다음 경우 감점하세요:
- 근거에 없는 내용을 사실처럼 말한 경우
- 종이 다른 동물 기준으로 조언한 경우
- 불필요하게 공포를 조장하거나 지나치게 안일한 경우

---

아래 형식의 STRICT JSON만 반환하세요:

{{
  "medical_score": <1~5>,
  "evidence_score": <1~5>,
  "medical_reason": "간단한 이유",
  "evidence_reason": "간단한 이유"
}}
"""




    response = llm.invoke(prompt).content

    try:
        return json.loads(response)
    except Exception:
        return {
            "medical_score": None,
            "evidence_score": None,
            "medical_reason": "parse_error",
            "evidence_reason": "parse_error",
        }