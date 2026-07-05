"""PDF text extraction helpers."""

from __future__ import annotations

from typing import BinaryIO

import fitz


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
