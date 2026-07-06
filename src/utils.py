"""Presentation helpers shared by the application."""

from __future__ import annotations

import re


def format_score(score: float) -> str:
    """Render a cosine similarity score consistently."""
    return f"{score:.3f}"


def clean_source_text(text: str) -> str:
    """Clean layout noise for evidence display without changing indexed text."""
    cleaned_lines = []
    for line in text.replace("\r", "\n").splitlines():
        clean = re.sub(r"[ \t]+", " ", line).strip()
        clean = re.sub(r"\.{4,}\s*\d*\s*$", "", clean).strip()
        if clean and (not cleaned_lines or clean != cleaned_lines[-1]):
            cleaned_lines.append(clean)
    return "\n".join(cleaned_lines)


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
            section = source.get("section_title") or "Unlabelled section"
            lines.append(
                f"{position}. {source['document_name']} - page {source['page_number']} - "
                f"{section} (similarity: {format_score(source['score'])}, "
                f"combined: {format_score(source.get('combined_score', source['score']))}, "
                f"{source['chunk_id']})"
            )
    else:
        lines.append("No source met the minimum similarity threshold.")
    return "\n".join(lines)
