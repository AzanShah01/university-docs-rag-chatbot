"""Word-aware text chunking with page-level metadata."""

from __future__ import annotations

import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.pdf_loader import is_low_value_text


def normalize_whitespace(text: str) -> str:
    """Normalize PDF text while preserving headings and paragraph boundaries."""
    normalized_lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        if clean_line:
            normalized_lines.append(clean_line)
    return "\n".join(normalized_lines)


def looks_like_section_heading(line: str) -> bool:
    """Detect numbered and short title-style section headings."""
    clean = re.sub(r"\s+", " ", line).strip()
    words = clean.split()
    if not clean or len(words) > 12 or len(clean) > 140:
        return False
    if clean.rstrip().endswith((".", "!", "?")) and not re.search(r"\.{4,}\s*\d*$", clean):
        return False
    if re.match(r"^chapter\s+\d+\b", clean, re.I):
        return True
    if re.match(r"^\d+(?:\.\d+)+\s+\S+", clean):
        return True
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not alpha_words:
        return False
    title_ratio = sum(word[:1].isupper() for word in alpha_words) / len(alpha_words)
    uppercase = clean.isupper() and len(alpha_words) <= 10
    return uppercase or (len(alpha_words) <= 8 and title_ratio >= 0.65)


def section_headings(text: str) -> list[tuple[int, str]]:
    """Return character offsets and cleaned titles for headings in page text."""
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        clean = re.sub(r"\s+", " ", line).strip()
        if looks_like_section_heading(clean):
            clean = re.sub(r"\.{4,}\s*\d*$", "", clean).strip()
            headings.append((offset, clean))
        offset += len(line)
    return headings


def compose_section_title(titles: list[str], inherited: str = "") -> str:
    """Keep chapter context when a chunk belongs to a numbered subsection."""
    if not titles:
        return inherited
    latest = titles[-1]
    if not re.match(r"^\d+(?:\.\d+)+\s+", latest):
        return latest

    parent = ""
    for candidate in reversed(titles[:-1]):
        if re.match(r"^chapter\s+\d+\b", candidate, re.I):
            parent = candidate
            break
        if re.match(r"^\d+\s+", candidate):
            parent = candidate
            break
    if not parent and inherited:
        inherited_parts = [part.strip() for part in inherited.split(" > ")]
        parent = next(
            (
                part for part in reversed(inherited_parts)
                if re.match(r"^chapter\s+\d+\b", part, re.I)
                or re.match(r"^\d+\s+", part)
            ),
            "",
        )
    return f"{parent} > {latest}" if parent else latest


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
    last_section_by_document: dict[str, str] = {}

    for page in pages:
        clean_text = normalize_whitespace(page.get("text", ""))
        document_name = page["document_name"]
        headings = section_headings(clean_text)
        inherited_section = last_section_by_document.get(document_name, "")
        word_matches = list(re.finditer(r"\S+", clean_text))
        if not word_matches:
            continue

        for start in range(0, len(word_matches), step):
            end = min(start + chunk_size, len(word_matches))
            if start >= end:
                break

            chunk_text = clean_text[word_matches[start].start() : word_matches[end - 1].end()]
            chunk_start = word_matches[start].start()
            heading_limit = word_matches[min(start + 60, end - 1)].end()
            matching_headings = [
                title for offset, title in headings if offset <= heading_limit
            ]
            section_title = compose_section_title(matching_headings, inherited_section)

            chunk_index = len(chunks)
            chunks.append(
                {
                    "text": chunk_text,
                    "document_name": document_name,
                    "page_number": page["page_number"],
                    "chunk_id": f"chunk-{chunk_index + 1}",
                    "section_title": section_title,
                    "low_value": bool(
                        page.get("low_value", False) or is_low_value_text(chunk_text)
                    ),
                }
            )
            if end >= len(word_matches):
                break

        if headings:
            last_section_by_document[document_name] = compose_section_title(
                [title for _, title in headings], inherited_section
            )

    return chunks
