# 근거 문서 인용정리
def build_citations(docs):
    citations = []
    for i, doc in enumerate(docs):
        citations.append({
            "id": i,
            "content": doc.page_content,

            "source_question": doc.metadata.get("question", "unknown"),

            "source_url": doc.metadata.get("url"),

            "source_title": doc.metadata.get("title")
        })
    return citations
