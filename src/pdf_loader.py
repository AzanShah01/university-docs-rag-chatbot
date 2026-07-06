"""PDF text extraction helpers."""

from __future__ import annotations

import re
from typing import BinaryIO

import fitz


LOW_VALUE_HEADINGS = {
    "table of contents",
    "contents",
    "list of figures",
    "list of tables",
    "declaration",
    "approval",
    "acknowledgement",
    "acknowledgements",
}


def is_low_value_text(text: str) -> bool:
    """Detect front matter and structure-only text without deleting it."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return True

    first_headings = {
        re.sub(r"[^a-z ]", "", line.lower()).strip()
        for line in lines[:8]
    }
    if first_headings & LOW_VALUE_HEADINGS:
        return True

    dotted_leaders = len(re.findall(r"\.{4,}\s*\d*", text))
    page_reference_lines = sum(
        bool(re.search(r"(?:\.{2,}|\s{3,})\s*\d{1,4}\s*$", line))
        for line in lines
    )
    numbered_heading_lines = sum(
        bool(re.match(r"^(?:chapter\s+)?\d+(?:\.\d+)*\s+.{2,80}(?:\s+\d+)?$", line, re.I))
        for line in lines
    )
    prose_lines = sum(
        len(line.split()) >= 12 and line.rstrip().endswith((".", "!", "?"))
        for line in lines
    )
    mostly_headings = (
        len(lines) >= 5
        and numbered_heading_lines / len(lines) >= 0.50
        and prose_lines <= 1
    )
    return dotted_leaders >= 3 or page_reference_lines >= 4 or mostly_headings


def extract_pdf_pages(pdf_file: BinaryIO) -> tuple[list[dict], list[str]]:
    """Extract readable pages from one uploaded PDF while retaining metadata."""
    document_name = getattr(pdf_file, "name", "uploaded_document.pdf")
    warnings: list[str] = []
    pages: list[dict] = []

    try:
        pdf_bytes = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
        if not pdf_bytes:
            return [], [f"{document_name} is empty and was skipped."]

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                return [], [f"{document_name} contains no pages and was skipped."]

            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = page.get_text("text").strip()
                if text:
                    pages.append(
                        {
                            "text": text,
                            "document_name": document_name,
                            "page_number": page_index + 1,
                            "low_value": is_low_value_text(text),
                        }
                    )

            if not pages:
                warnings.append(
                    f"No readable text was found in {document_name}. It may be scanned, "
                    "image-only, empty, or protected."
                )
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        warnings.append(f"Could not read {document_name}: {exc}")
    except Exception as exc:
        warnings.append(f"Unexpected error while reading {document_name}: {exc}")

    return pages, warnings


def load_pdf_files(uploaded_files: list[BinaryIO]) -> tuple[list[dict], list[str]]:
    """Extract pages from multiple uploads without failing the entire batch."""
    all_pages: list[dict] = []
    all_warnings: list[str] = []
    for uploaded_file in uploaded_files:
        pages, warnings = extract_pdf_pages(uploaded_file)
        all_pages.extend(pages)
        all_warnings.extend(warnings)
    return all_pages, all_warnings
