from typing import List, Dict

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import CrossEncoder

from config import *

# ingest.py의 animal detector 재사용
from ingest import detect_animal

# 🔥 증상 분류기 import
from categorize import categorize_text


# =========================
# Global models (1회 로드)
# =========================

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

rewrite_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================
# Query rewriting (history-aware)
# =========================

def rewrite_query(query: str, history: List[Dict[str, str]] = None) -> str:
    """
    history가 있으면 최근 대화 맥락을 포함해 query를 재작성
    """

    history_text = ""
    if history:
        recent = history[-2:]  # 🔑 최근 2턴만 사용
        history_text = "\n".join(
            f"사용자: {h['user']}\n답변: {h['assistant']}"
            for h in recent
        )

    prompt = f"""
다음은 반려동물 보호자의 질문이다.
검색을 위해 보호자의 궁금증과 상황을 더 명확히 드러내는 질문으로 바꿔라.

이전 대화 맥락:
{history_text if history_text else "없음"}

규칙:
- 이전 대화 맥락이 있다면 반드시 반영할 것
- 보호자가 느낀 증상의 변화(증가, 감소, 평소와 다름)를 포함할 것
- 보호자가 궁금해하는 점(정상인지, 병원에 가야 하는지 등)을 질문 형태로 확장할 것
- 판단, 조언, 권장 표현은 사용하지 말 것
- 원인을 단정하지 말 것
- 서너 문장으로 작성할 것
- 전체를 질문 형태로 유지할 것

원문 질문: {query}
변환:
"""
    return rewrite_llm.invoke(prompt).content.strip()


# =========================
# Retrieval (멀티턴 대응)
# =========================

def retrieve_docs(
    query: str,
    history: List[Dict[str, str]] = None,
    k: int = 3,
    fetch_k: int = 50,
):
    """
    query + history
    → animal 판단 (query only)
    → symptom category 판단 (query only)
    → history-aware query rewriting
    → Pinecone retrieval (filter)
    → cross-encoder rerank
    """

    # ===============================
    # 0️⃣ animal 판단 (현재 질문 기준)
    # ===============================
    animal = detect_animal(question=query)
    print(f"[DEBUG] detected animal: {animal}")

    # ===============================
    # 0️⃣-2 symptom category 판단
    # ===============================
    symptom_category, symptom_conf = categorize_text(query)
    print(f"[DEBUG] symptom_category={symptom_category}, conf={symptom_conf:.3f}")

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    # ===============================
    # 1️⃣ Query rewriting (🔥 history 반영)
    # ===============================
    rewritten_query = rewrite_query(query, history)

    print("\n=== QUERY REWRITE DEBUG ===")
    print("ORIGINAL :", query)
    print("HISTORY  :", history[-2:] if history else "None")
    print("REWRITTEN:", rewritten_query)
    print("==========================\n")

    # ===============================
    # 2️⃣ Pinecone filter 구성
    # ===============================
    pinecone_filter = {}

    # animal filter
    if animal in ("cat", "dog"):
        pinecone_filter["animal"] = {"$in": [animal, "unknown"]}

    # symptom filter (confidence 기준)
    if symptom_conf >= 0.5 and symptom_category != "미분류":
        pinecone_filter["symptom_category"] = symptom_category

    print(f"[DEBUG] pinecone_filter = {pinecone_filter}")

    # ===============================
    # 3️⃣ Pinecone recall
    # ===============================
    docs = vectorstore.similarity_search(
        rewritten_query,
        k=fetch_k,
        filter=pinecone_filter if pinecone_filter else None
    )

    if not docs:
        print("[WARN] Pinecone returned 0 documents.")
        return []

    # ===============================
    # 4️⃣ Cross-Encoder reranking
    # ===============================
    pairs = [
        (rewritten_query, d.page_content)
        for d in docs
    ]

    scores = cross_encoder.predict(pairs)

    reranked = []
    for doc, score in zip(docs, scores):
        penalty = 0.0

        if doc.metadata.get("animal") == "unknown":
            penalty += 0.3

        if symptom_conf >= 0.5:
            if doc.metadata.get("symptom_category") != symptom_category:
                penalty += 0.5

        reranked.append((doc, score - penalty))

    reranked = sorted(
        reranked,
        key=lambda x: x[1],
        reverse=True
    )

    # ===============================
    # 5️⃣ Debug 출력
    # ===============================
    print("\n================ RERANK DEBUG ====================")
    for i, (doc, score) in enumerate(reranked[:k]):
        print(f"[RERANK {i}] score={score:.4f}")
        print("ANIMAL:", doc.metadata.get("animal"))
        print("SYMPTOM:", doc.metadata.get("symptom_category"))
        print("URL:", doc.metadata.get("url"))
        print("QUESTION:", doc.metadata.get("question"))
        print("CONTENT (HEAD):")
        print(doc.page_content[:300])
        print("------------------------------------------------")
    print("=================================================\n")

    return [doc for doc, _ in reranked[:k]]
