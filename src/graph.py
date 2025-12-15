from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from observe.trace_utils import traced_node

from rag.retriever import retrieve_docs
from rag.citation import build_citations
from rag.generator import generate_answer
from safety.guardrail import apply_guardrail
from evaluation.judge import judge_answer
from postprocess import (
    confidence_level,
    extract_urls,
)

class GraphState(TypedDict):
    question: str
    history: List[Dict[str, str]]  # 🔥 추가

    docs: list
    citations: List[Dict[str, Any]]

    answer: str
    evaluation: Dict[str, Any]

    confidence: str
    evidence_urls: List[str]



def build_graph():

    graph = StateGraph(GraphState)

    # -------------------------
    # Retrieve
    # -------------------------
    graph.add_node(
        "retrieve",
        traced_node(
            "retrieve",
            lambda s: {
                **s,
                "docs": retrieve_docs(s["question"],
                                      history=s.get("history", [])),
            },
        ),
    )

    # -------------------------
    # Citation
    # -------------------------
    graph.add_node(
        "cite",
        traced_node(
            "cite",
            lambda s: {
                **s,
                "citations": build_citations(s["docs"]),
            },
        ),
    )

    # -------------------------
    # Generate answer (LLM)
    # -------------------------
    graph.add_node(
        "generate",
        traced_node(
            "generate",
            lambda s: {
                **s,
                "answer": generate_answer(
                    question=s["question"],
                    history=s.get("history", []),
                    citations=s["citations"],
                ),
            },
        ),
    )

    # -------------------------
    # Safety guardrail
    # -------------------------
    graph.add_node(
        "safety",
        traced_node(
            "safety",
            lambda s: {
                **s,
                "answer": apply_guardrail(s["answer"]),
            },
        ),
    )

    # -------------------------
    # LLM-as-Judge
    # -------------------------
    graph.add_node(
        "judge",
        traced_node(
            "judge",
            lambda s: {
                **s,
                "evaluation": judge_answer(
                    question=s["question"],
                    answer=s["answer"],
                    citations=s["citations"],
                ),
            },
        ),
    )

    # -------------------------
    # Postprocess (confidence + URLs)
    # -------------------------
    graph.add_node(
        "postprocess",
        traced_node(
            "postprocess",
            lambda s: {
                **s,
                "confidence": confidence_level(
                    medical_score=s["evaluation"].get("medical_score"),
                    evidence_score=s["evaluation"].get("evidence_score"),
                    has_evidence=len(extract_urls(s["citations"])) > 0,
                ),
                "evidence_urls": extract_urls(s["citations"]),
            },
        ),
    )

    # ----------------------------------------------------------------
    # 3. Edges
    # ----------------------------------------------------------------
    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "cite")
    graph.add_edge("cite", "generate")
    graph.add_edge("generate", "safety")
    graph.add_edge("safety", "judge")
    graph.add_edge("judge", "postprocess")
    graph.add_edge("postprocess", END)

    return graph.compile()