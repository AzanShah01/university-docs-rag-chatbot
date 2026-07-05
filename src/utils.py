"""Presentation helpers shared by the application."""

from __future__ import annotations


def format_score(score: float) -> str:
    """Render a cosine similarity score consistently."""
    return f"{score:.3f}"


def format_answer_download(result: dict) -> str:
    """Build a plain-text answer with traceable source references."""
    lines = [
        "University Docs RAG Chatbot",
        "",
        f"Question: {result['question']}",
        "",
        "Answer:",
        result["answer"],
        "",
        f"Confidence: {result.get('confidence', 'Not available')}",
        "",
        "Sources:",
    ]
    if result["sources"]:
        for position, source in enumerate(result["sources"], start=1):
            lines.append(
                f"{position}. {source['document_name']} - page {source['page_number']} "
                f"(similarity: {format_score(source['score'])}, {source['chunk_id']})"
            )
    else:
        lines.append("No source met the minimum similarity threshold.")
    return "\n".join(lines)
