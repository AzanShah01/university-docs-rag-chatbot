"""Word-aware text chunking with page-level metadata."""

from __future__ import annotations

import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def normalize_whitespace(text: str) -> str:
    """Normalize PDF text while preserving headings and paragraph boundaries."""
    normalized_lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        if clean_line:
            normalized_lines.append(clean_line)
    return "\n".join(normalized_lines)


def split_documents(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split each page into overlapping word chunks and retain its source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[dict] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        clean_text = normalize_whitespace(page.get("text", ""))
        word_matches = list(re.finditer(r"\S+", clean_text))
        if not word_matches:
            continue

        for start in range(0, len(word_matches), step):
            end = min(start + chunk_size, len(word_matches))
            if start >= end:
                break

            chunk_text = clean_text[word_matches[start].start() : word_matches[end - 1].end()]

            chunk_index = len(chunks)
            chunks.append(
                {
                    "text": chunk_text,
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "chunk_id": f"chunk-{chunk_index + 1}",
                }
            )
            if end >= len(word_matches):
                break

    return chunks
