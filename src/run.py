from graph import build_graph
import json

from rag.retriever import retrieve_docs


def debug_retriever(query: str):
    docs = retrieve_docs(query)
    print("\n=== RETRIEVER DEBUG ===")
    for i, d in enumerate(docs):
        print(f"[DOC {i}]")
        print("URL:", d.metadata.get("url"))
        print("QUESTION:", d.metadata.get("question"))
        print("CONTENT HEAD:")
        print(d.page_content[:300])
        print("----------------------")
    print("======================\n")


def main():
    app = build_graph()

    print("🐾 Pet Medical RAG CLI")
    print("질문을 입력하세요. 종료하려면 'exit' 또는 'quit' 입력\n")

    while True:
        question = input("Q> ").strip()

        if not question or question.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break

        # 🔍 retriever 단독 디버깅 (원하면 주석 해제)
        # debug_retriever(question)

        result = app.invoke({
            "question": question
        })

        print("\n=== 답변 ===")
        print(result["answer"])

        print("\n=== 확신도 ===")
        print(result["confidence"])

        print("\n=== 근거 URL ===")
        for url in result.get("evidence_urls", []):
            print("-", url)

        if "evaluation" in result:
            print("\n=== 평가 점수 ===")
            print(json.dumps(result["evaluation"], indent=2, ensure_ascii=False))

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
