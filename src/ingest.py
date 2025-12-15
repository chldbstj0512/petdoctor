import pandas as pd

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
)

# 🔥 증상 분류기 import
from categorize import categorize_text


# =========================
# 1️⃣ 동물 종류 판단 (가중치 기반)
# =========================

ANIMAL_KEYWORDS = {
    "dog": [
        "강아지", "반려견", "댕댕이",
        "보더콜리", "푸들", "말티즈", "시바",
        "산책", "목줄", "배변훈련",
    ],
    "cat": [
        "고양이", "길냥이", "냥이", "반려묘", "야옹이",
        "캣", "스크래쳐", "모래", "화장실",
        "캣타워",
    ],
}


def detect_animal(
    question: str = "",
    title: str = "",
    min_hits: int = 1,
    ratio_threshold: float = 0.4,
) -> str:
    """
    동물 종류 판별 (빈도 기반)
    - dog / cat 키워드 카운트 비교
    - 지배적인 쪽만 확정
    """

    text = f"{title} {question}"

    scores = {}
    for animal, keywords in ANIMAL_KEYWORDS.items():
        scores[animal] = sum(text.count(kw) for kw in keywords)

    best_animal = max(scores, key=scores.get)
    best_score = scores[best_animal]
    total_score = sum(scores.values())

    # 키워드가 거의 없는 경우
    if best_score < min_hits or total_score == 0:
        return "unknown"

    confidence = best_score / total_score

    # 충분히 지배적인 경우만 확정
    if confidence >= ratio_threshold:
        return best_animal

    return "unknown"


# =========================
# 2️⃣ CSV → Pinecone Ingest
# =========================

def ingest_csv(csv_path="/home/ys0660/happycat/data/data.csv"):
    # 1️⃣ CSV 로드
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["answer", "question"])
    df = df.fillna("")

    # sanity check
    for i, row in df.iterrows():
        if pd.isna(row.get("url")) or pd.isna(row.get("answer_type")):
            print("BAD ROW:", i, row)
            break

    # 2️⃣ Pinecone client
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    # 3️⃣ Embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    # 4️⃣ Document 생성
    docs = []

    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        title = str(row.get("title", ""))
        answer = str(row.get("answer_clean", ""))
        url = str(row.get("url", ""))
        answer_type = str(row.get("answer_type", "unknown"))

        # 🔹 동물 종류 판단 (가중치 기반)
        animal = detect_animal(
            question=question,
            title=title,
        )

        # 🔥 증상 카테고리 분류
        symptom_category, symptom_confidence = categorize_text(question)

        # Q + A 결합 (retrieval 대상)
        page_content = f"Q: {question}\nA: {answer}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    # 기존 필드
                    "question": question,
                    "title": title,
                    "url": url,
                    "answer_type": answer_type,
                    "animal": animal,

                    # 신규 필드
                    "symptom_category": symptom_category,
                    "symptom_confidence": symptom_confidence,
                }
            )
        )

    print(f"Loaded {len(docs)} documents from CSV")

    # 5️⃣ VectorStore 연결 (기존 index 사용)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    # 6️⃣ 업로드
    vectorstore.add_documents(docs)

    print("✅ Pinecone ingestion completed.")


if __name__ == "__main__":
    ingest_csv()
