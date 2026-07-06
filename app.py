"""Streamlit entry point for the University Docs RAG Chatbot."""

from __future__ import annotations

import hashlib
import html
import os
from typing import Any

import streamlit as st

from src.config import EMBEDDING_MODEL, MIN_SIMILARITY_SCORE, TOP_K
from src.utils import clean_source_text, format_answer_download, format_score

st.set_page_config(
    page_title="UniDocs Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAFETY_DISCLAIMER = (
    "Answers are extracted from uploaded documents and may not be 100% accurate. "
    "Verify important information with the university."
)
EXAMPLE_QUESTIONS = (
    "What is this document about?",
    "What are the dataset collection details?",
    "What are the system limitations?",
    "What are the future improvements?",
)
PROCESSING_STEPS = (
    "Extracting PDF text",
    "Splitting text into chunks",
    "Loading embedding model",
    "Building FAISS search index",
    "Ready for questions",
)


def apply_styles() -> None:
    """Apply a restrained visual system without replacing native controls."""
    st.markdown(
        """
        <style>
        :root { --navy: #0F172A; --ink: #0F172A; --slate: #334155;
            --muted: #64748B; --brand: #2563EB; --brand-dark: #1D4ED8;
            --surface: #FFFFFF; --background: #F8FAFC; --line: #E2E8F0; }
        .stApp { background: var(--background); color: var(--ink); }
        .block-container { max-width: 1120px; padding: 2rem 2.4rem 4rem; }
        [data-testid="stSidebar"] { background: var(--navy); border-right: 1px solid #1E293B; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #F8FAFC !important; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #CBD5E1 !important; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
        .hero { padding: 1.55rem 1.7rem; color: var(--ink); background: white;
            border: 1px solid var(--line); border-left: 5px solid var(--brand);
            border-radius: 1rem; margin-bottom: 1.2rem;
            box-shadow: 0 10px 30px rgba(23, 27, 39, .06); }
        .hero-kicker { font-size: .75rem; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; color: var(--brand); }
        .hero h1 { color: var(--ink); font-size: 2rem; margin: .45rem 0; }
        .hero p { margin: 0; max-width: 760px; color: var(--muted); }
        .brand { font-size: 1.3rem; font-weight: 750; color: #F8FAFC; }
        .brand-mark { display: inline-flex; width: 2rem; height: 2rem; align-items: center;
            justify-content: center; border-radius: .6rem; margin-right: .5rem;
            color: white; background: var(--brand); }
        .eyebrow { color: #93C5FD; font-size: .72rem; font-weight: 800;
            letter-spacing: .09em; text-transform: uppercase; margin: 1.25rem 0 .5rem;
            padding-bottom: .35rem; border-bottom: 1px solid #334155; }
        .status-dot { display: inline-block; width: .55rem; height: .55rem;
            border-radius: 50%; margin-right: .45rem; background: #f59e0b; }
        .status-dot.ready { background: #16a34a; }
        .status-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .45rem;
            margin-top: .7rem; }
        .status-item { background: #1E293B; border: 1px solid #334155; border-radius: .65rem;
            padding: .55rem .35rem; text-align: center; }
        .status-value { color: #F8FAFC; font-size: 1rem; font-weight: 800; line-height: 1.2; }
        .status-label { color: #CBD5E1; font-size: .64rem; text-transform: uppercase;
            letter-spacing: .06em; margin-top: .2rem; }
        .privacy-note { color: #CBD5E1; font-size: .82rem; line-height: 1.55; }
        .stButton > button, .stDownloadButton > button { border-radius: .65rem; font-weight: 650; }
        [data-testid="stSidebar"] .stButton > button { min-height: 2.8rem; }
        [data-testid="stFileUploaderDropzone"] { background: #f7f8fb; border-radius: .8rem;
            border-color: #d9dee8; }
        [data-testid="stFileUploaderDropzone"] button { background: #fff; color: var(--ink);
            border: 1px solid #d9dee8; }
        [data-testid="stSidebar"] details { background: #1E293B; border-color: #334155; }
        [data-testid="stSidebar"] .stButton > button { background: #1E293B; color: #F8FAFC;
            border: 1px solid #475569; }
        [data-testid="stSidebar"] .stButton > button:hover { background: #2563EB;
            border-color: #60A5FA; color: #fff; }
        .workspace-title { color: var(--ink); font-size: 1rem; font-weight: 750;
            margin-bottom: .15rem; }
        .workspace-copy { color: var(--muted); font-size: .86rem; margin-bottom: .7rem; }
        .ready-card { background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: .9rem;
            padding: .9rem 1rem; margin: .2rem 0 .75rem; }
        .ready-heading { color: #1E3A8A; font-size: 1rem; font-weight: 800; margin-bottom: .65rem; }
        .ready-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .55rem; }
        .ready-item { background: #fff; border: 1px solid #DBEAFE; border-radius: .65rem;
            padding: .55rem .7rem; }
        .ready-value { color: var(--ink); font-size: .95rem; font-weight: 800; }
        .ready-label { color: var(--muted); font-size: .66rem; letter-spacing: .05em;
            text-transform: uppercase; margin-top: .15rem; }
        .processing-success { color: #166534; background: #F0FDF4; border: 1px solid #BBF7D0;
            border-radius: .65rem; padding: .55rem .75rem; font-size: .84rem; margin: .45rem 0; }
        .processing-error { color: #991B1B; background: #FEF2F2; border: 1px solid #FECACA;
            border-radius: .65rem; padding: .55rem .75rem; font-size: .84rem; margin: .45rem 0; }
        [data-testid="stProgress"] > div > div { background-color: var(--brand); }
        .answer-label { color: var(--brand); font-size: .7rem; font-weight: 800;
            letter-spacing: .09em; text-transform: uppercase; margin: .1rem 0 .45rem; }
        .confidence-pill { display: inline-flex; align-items: center; border-radius: 999px;
            padding: .22rem .6rem; font-size: .72rem; font-weight: 750; margin-bottom: .65rem; }
        .confidence-high { color: #166534; background: #dcfce7; }
        .confidence-medium { color: #1E40AF; background: #DBEAFE; }
        .confidence-low { color: #475569; background: #E2E8F0; }
        .question-label, .answer-section-label { color: var(--muted); font-size: .68rem;
            font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
        .question-text { color: var(--ink); font-size: 1rem; font-weight: 700;
            margin: .2rem 0 .7rem; }
        .answer-section-label { margin: .2rem 0 .3rem; }
        .weak-note { color: #475569; background: #F1F5F9; border-left: 3px solid #94A3B8;
            border-radius: .4rem; padding: .5rem .7rem; font-size: .8rem; margin: .55rem 0; }
        .how-steps { list-style: none; counter-reset: steps; margin: .3rem 0 1rem; padding: 0; }
        .how-steps li { counter-increment: steps; display: flex; align-items: center; gap: .65rem;
            color: #E2E8F0; font-size: .85rem; margin: .55rem 0; }
        .how-steps li::before { content: counter(steps); display: inline-flex; align-items: center;
            justify-content: center; min-width: 1.55rem; height: 1.55rem; border-radius: 50%;
            background: #1D4ED8; color: #fff; font-size: .72rem; font-weight: 800; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.answer-label) { background: #fff;
            border-color: var(--line); box-shadow: 0 6px 20px rgba(15, 23, 42, .04); }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.answer-label) [data-testid="stVerticalBlock"] {
            gap: .55rem; }
        @media (max-width: 700px) {
            .ready-grid { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> Any:
    """Load from the local cache first, downloading only when it is missing."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except (OSError, ValueError):
        return SentenceTransformer(EMBEDDING_MODEL)


def files_fingerprint(uploaded_files: list) -> str:
    """Create a stable identifier for the current upload selection."""
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8", errors="ignore"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def initialize_state() -> None:
    defaults = {
        "rag_pipeline": None,
        "processed_chunks": [],
        "processed_fingerprint": None,
        "processing_summary": None,
        "processing_warnings": [],
        "messages": [],
        "uploader_key": 0,
        "question_key": 0,
        "show_examples": True,
        "processing_state": {
            "status": "idle",
            "completed": 0,
            "active": None,
            "failed": None,
            "message": "",
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation() -> None:
    """Clear chat history while keeping the processed document index."""
    st.session_state.messages = []
    st.session_state.question_key += 1
    st.session_state.show_examples = True


def documents_are_ready(uploaded_files: list) -> bool:
    """Return whether the current uploader selection matches the ready index."""
    if not uploaded_files or st.session_state.rag_pipeline is None:
        return False
    return bool(
        st.session_state.rag_pipeline.is_ready
        and files_fingerprint(uploaded_files) == st.session_state.processed_fingerprint
    )


def update_processing_state(
    *,
    status: str,
    completed: int,
    active: int | None,
    message: str,
    failed: int | None = None,
    tracker: dict | None = None,
) -> None:
    """Persist processing progress and update visible tracker widgets."""
    state = {
        "status": status,
        "completed": completed,
        "active": active,
        "failed": failed,
        "message": message,
    }
    st.session_state.processing_state = state
    if tracker:
        tracker["progress"].progress(completed / len(PROCESSING_STEPS))
        tracker["message"].markdown(f"**{message}**")
        tracker["steps"].markdown(processing_steps_markdown(state))


def processing_steps_markdown(state: dict) -> str:
    """Build the completed/running/pending processing checklist."""
    lines = []
    for index, label in enumerate(PROCESSING_STEPS):
        if state.get("failed") == index:
            indicator = "❌"
        elif index < state.get("completed", 0):
            indicator = "✅"
        elif state.get("active") == index:
            indicator = "⏳"
        else:
            indicator = "⬜"
        lines.append(f"{indicator} {label}")
    return "  \n".join(lines)


def render_processing_details(expanded: bool) -> dict:
    """Render the persisted tracker and return widgets for live updates."""
    state = st.session_state.processing_state
    with st.expander("Processing details", expanded=expanded):
        progress = st.progress(state.get("completed", 0) / len(PROCESSING_STEPS))
        message = st.empty()
        steps = st.empty()
        message.markdown(f"**{state.get('message') or 'Waiting to process documents.'}**")
        steps.markdown(processing_steps_markdown(state))
    return {"progress": progress, "message": message, "steps": steps}


def reset_app() -> None:
    """Remove uploaded-document state and start a fresh uploader widget."""
    st.session_state.rag_pipeline = None
    st.session_state.processed_chunks = []
    st.session_state.processed_fingerprint = None
    st.session_state.processing_summary = None
    st.session_state.processing_warnings = []
    st.session_state.messages = []
    st.session_state.show_examples = True
    st.session_state.uploader_key += 1
    st.session_state.question_key += 1
    st.session_state.processing_state = {
        "status": "idle",
        "completed": 0,
        "active": None,
        "failed": None,
        "message": "",
    }


def process_documents(uploaded_files: list, tracker: dict) -> list[str]:
    """Extract, chunk, embed, and index uploaded PDFs with visible progress."""
    # Heavy libraries load here so the interface can render immediately.
    from src.pdf_loader import load_pdf_files
    from src.rag_pipeline import RAGPipeline
    from src.text_splitter import split_documents

    update_processing_state(
        status="running",
        completed=0,
        active=0,
        message="Currently extracting text from uploaded PDFs...",
        tracker=tracker,
    )
    pages, warnings = load_pdf_files(uploaded_files)
    st.session_state.processing_warnings = warnings
    if not pages:
        raise ValueError("No readable text was found. Scanned PDFs need OCR.")

    update_processing_state(
        status="running",
        completed=1,
        active=1,
        message="Currently splitting text into searchable chunks...",
        tracker=tracker,
    )
    chunks = split_documents(pages)
    if not chunks:
        raise ValueError("The extracted text could not be split into usable chunks.")

    update_processing_state(
        status="running",
        completed=2,
        active=2,
        message="Currently loading the local embedding model...",
        tracker=tracker,
    )
    model = load_embedding_model()

    update_processing_state(
        status="running",
        completed=3,
        active=3,
        message="Currently building the FAISS search index...",
        tracker=tracker,
    )
    pipeline = RAGPipeline(model=model)
    pipeline.index_chunks(chunks)

    update_processing_state(
        status="running",
        completed=4,
        active=4,
        message="Finalizing the document workspace...",
        tracker=tracker,
    )

    st.session_state.rag_pipeline = pipeline
    st.session_state.processed_chunks = chunks
    st.session_state.processed_fingerprint = files_fingerprint(uploaded_files)
    st.session_state.processing_summary = {
        "documents": len({page["document_name"] for page in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
    }
    st.session_state.processing_warnings = warnings
    st.session_state.messages = []
    st.session_state.show_examples = True
    update_processing_state(
        status="complete",
        completed=5,
        active=None,
        message="Documents are ready for questions.",
        tracker=tracker,
    )
    return warnings


def render_sources(sources: list[dict], compact: bool = False) -> None:
    visible_sources = sources[:3]
    if not visible_sources:
        return
    source_container = (
        st.popover(f"Source evidence ({len(visible_sources)})")
        if compact
        else st.expander(f"Source evidence ({len(visible_sources)})")
    )
    with source_container:
        for position, source in enumerate(visible_sources, start=1):
            st.markdown(f"**Source {position}**")
            section_title = source.get("section_title") or "Unlabelled section"
            st.markdown(f"**{section_title}**")
            st.write(source["document_name"])
            st.caption(
                f"Page {source['page_number']}  |  "
                f"Similarity {format_score(source['score'])}  |  "
                f"Combined {format_score(source.get('combined_score', source['score']))}  |  "
                f"{source['chunk_id']}"
            )
            st.text(clean_source_text(source["text"]))
            if position < len(visible_sources):
                st.divider()


def render_answer_card(
    question: str,
    result: dict,
    message_index: int,
    compact_sources: bool = False,
) -> None:
    """Render a compact question, confidence, answer, and evidence card."""
    confidence = result.get("confidence", "Low confidence")
    confidence_class = confidence.split()[0].lower()
    confidence_score = result.get("confidence_score", 0.0)
    with st.container(border=True):
        st.markdown('<div class="answer-label">Document answer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="question-label">Question</div>'
            f'<div class="question-text">{html.escape(question)}</div>'
            f'<span class="confidence-pill confidence-{confidence_class}">'
            f'{confidence} · {confidence_score:.2f}</span>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="answer-section-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(result["answer"])
        if result.get("guidance"):
            st.markdown(
                f'<div class="weak-note">{html.escape(result["guidance"])}</div>',
                unsafe_allow_html=True,
            )
        render_sources(result.get("sources", []), compact=compact_sources)
        st.download_button(
            "Download answer",
            data=format_answer_download(result),
            file_name="unidocs_answer.txt",
            mime="text/plain",
            key=f"download-{message_index}",
        )


def conversation_turns() -> list[dict]:
    """Pair stored user messages with their following assistant results."""
    turns: list[dict] = []
    pending_question = ""
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            pending_question = message["content"]
        elif message["role"] == "assistant" and pending_question:
            turns.append(
                {
                    "question": pending_question,
                    "result": message["result"],
                    "message_index": index,
                }
            )
            pending_question = ""
    return turns


def render_document_controls(show_reset: bool) -> tuple[list, bool]:
    """Render upload/replace controls used by both workspace states."""
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files stay in this local session and are not sent to an AI API.",
        key=f"pdf-uploader-{st.session_state.uploader_key}",
    )
    process_clicked = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
        key="process-documents",
    )
    st.caption(
        "First use only: loading the local embedding model may take a little longer. "
        "It is cached for later processing and questions."
    )
    if show_reset:
        st.button(
            "Clear uploaded documents and reset app",
            use_container_width=True,
            on_click=reset_app,
            key="reset-main",
        )
    return uploaded_files, process_clicked


def render_ready_summary(uploaded_files: list) -> None:
    """Render the compact processed-document summary above Ask Questions."""
    summary = st.session_state.processing_summary or {}
    st.markdown(
        '<div class="ready-card"><div class="ready-heading">Documents ready</div>'
        '<div class="ready-grid">'
        f'<div class="ready-item"><div class="ready-value">{len(uploaded_files)}</div>'
        '<div class="ready-label">PDFs</div></div>'
        f'<div class="ready-item"><div class="ready-value">{summary.get("pages", 0)}</div>'
        '<div class="ready-label">Pages extracted</div></div>'
        f'<div class="ready-item"><div class="ready-value">{summary.get("chunks", 0)}</div>'
        '<div class="ready-label">Chunks</div></div>'
        '<div class="ready-item"><div class="ready-value">Ready</div>'
        '<div class="ready-label">Index status</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_sidebar(uploaded_files: list) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">U</span>UniDocs</div>',
            unsafe_allow_html=True,
        )
        st.caption("Local document assistant")
        st.markdown('<div class="eyebrow">Library status</div>', unsafe_allow_html=True)
        summary = st.session_state.processing_summary
        current = files_fingerprint(uploaded_files) if uploaded_files else None
        is_ready = bool(
            summary
            and st.session_state.rag_pipeline is not None
            and current == st.session_state.processed_fingerprint
        )
        chunk_count = len(st.session_state.processed_chunks) if is_ready else 0
        index_status = "Ready" if is_ready else "Not ready"
        st.markdown(
            '<div class="status-grid">'
            f'<div class="status-item"><div class="status-value">{len(uploaded_files)}</div>'
            '<div class="status-label">PDFs</div></div>'
            f'<div class="status-item"><div class="status-value">{chunk_count}</div>'
            '<div class="status-label">Chunks</div></div>'
            f'<div class="status-item"><div class="status-value">{index_status}</div>'
            '<div class="status-label">Index</div></div></div>',
            unsafe_allow_html=True,
        )
        if summary and is_ready:
            st.caption(
                f"{summary['documents']} document(s) · {summary['pages']} readable page(s) · "
                f"{summary['chunks']} passages"
            )
        elif summary:
            st.warning("Your file selection changed. Process it again to update the index.")

        st.button(
            "Clear uploaded documents and reset app",
            use_container_width=True,
            disabled=not uploaded_files and st.session_state.rag_pipeline is None,
            on_click=reset_app,
            key="reset-sidebar",
        )

        st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ol class="how-steps">
              <li>Upload PDFs</li>
              <li>Process documents</li>
              <li>Ask a question</li>
              <li>Verify source evidence</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Privacy & offline use"):
            st.markdown(
                '<div class="privacy-note">PDF text, embeddings, search, and answers stay '
                'on this computer. The first use downloads the embedding model; after it is '
                'cached, processing works offline. No API key or cloud AI is used.</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Supported documents"):
            st.caption(
                "Text-based PDFs work best. Image-only scans require OCR, which is not "
                "included. Complex tables and columns may extract imperfectly."
            )


def main() -> None:
    apply_styles()
    initialize_state()
    st.markdown(
        """
        <div class="hero">
          <span class="hero-kicker">Private · Source-grounded · Local</span>
          <h1>Ask your university documents</h1>
          <p>Find policies, requirements, dates, and procedures in your PDFs—with the
          original page evidence beside every answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploader_widget_key = f"pdf-uploader-{st.session_state.uploader_key}"
    existing_files = st.session_state.get(uploader_widget_key, []) or []
    ready_before_controls = documents_are_ready(existing_files)

    if ready_before_controls:
        render_ready_summary(existing_files)
        with st.expander("Manage uploaded documents", expanded=False):
            uploaded_files, process_clicked = render_document_controls(show_reset=True)
    else:
        with st.expander("Document workspace", expanded=True):
            st.markdown(
                '<div class="workspace-copy">Upload text-based PDFs, then build a local '
                'search index.</div>',
                unsafe_allow_html=True,
            )
            uploaded_files, process_clicked = render_document_controls(
                show_reset=bool(existing_files or st.session_state.rag_pipeline)
            )

    render_sidebar(uploaded_files)

    if process_clicked:
        update_processing_state(
            status="running",
            completed=0,
            active=0,
            message="Currently extracting text from uploaded PDFs...",
        )
        tracker = render_processing_details(expanded=True)
        try:
            process_documents(uploaded_files, tracker)
        except Exception as exc:
            state = st.session_state.processing_state
            failed_step = state.get("active")
            failed_label = (
                PROCESSING_STEPS[failed_step]
                if isinstance(failed_step, int)
                else "document processing"
            )
            update_processing_state(
                status="error",
                completed=state.get("completed", 0),
                active=None,
                failed=failed_step,
                message=f"Processing failed during {failed_label.lower()}: {exc}",
                tracker=tracker,
            )
            st.session_state.rag_pipeline = None
            st.session_state.processed_chunks = []
            st.session_state.processed_fingerprint = None
            st.session_state.processing_summary = None
            st.rerun()
        else:
            st.rerun()

    is_ready = documents_are_ready(uploaded_files)
    processing_state = st.session_state.processing_state
    if processing_state["status"] == "complete" and is_ready:
        st.markdown(
            '<div class="processing-success">Documents processed successfully. '
            'You can now ask questions.</div>',
            unsafe_allow_html=True,
        )
        render_processing_details(expanded=False)
    elif processing_state["status"] == "error":
        st.markdown(
            f'<div class="processing-error">{html.escape(processing_state["message"])}</div>',
            unsafe_allow_html=True,
        )
        render_processing_details(expanded=True)

    for warning in st.session_state.processing_warnings:
        st.warning(warning)

    if not uploaded_files:
        st.info("Upload one or more PDFs above to begin.")
    elif not is_ready:
        st.info("Select **Process documents** above to prepare your files.")

    title_col, clear_col = st.columns([3.2, 1])
    with title_col:
        st.subheader("Ask Questions")
        st.caption("Ask a focused question and verify the answer against its source evidence.")
    with clear_col:
        st.button(
            "Clear conversation",
            use_container_width=True,
            disabled=not st.session_state.messages,
            on_click=clear_conversation,
        )

    with st.container(border=True):
        with st.form("question-form", clear_on_submit=True):
            input_col, ask_col = st.columns([5, 1])
            with input_col:
                typed_question = st.text_input(
                    "Question",
                    placeholder="Ask a question about the processed documents...",
                    label_visibility="collapsed",
                    disabled=not is_ready,
                    key=f"question-{st.session_state.question_key}",
                )
            with ask_col:
                submitted = st.form_submit_button(
                    "Ask",
                    type="primary",
                    use_container_width=True,
                    disabled=not is_ready,
                )

    selected_example = None
    if is_ready:
        with st.expander("Example questions"):
            example_columns = st.columns(2)
            for index, example in enumerate(EXAMPLE_QUESTIONS):
                with example_columns[index % 2]:
                    if st.button(example, key=f"example-{index}", use_container_width=True):
                        selected_example = example

    question = selected_example or (typed_question if submitted else None)
    if question:
        st.session_state.question_key += 1
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Finding the most relevant answer..."):
                result = st.session_state.rag_pipeline.answer_question(
                    question=question,
                    top_k=TOP_K,
                    min_similarity=MIN_SIMILARITY_SCORE,
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "result": result}
            )
            st.rerun()
        except Exception as exc:
            st.session_state.messages.pop()
            st.error(f"Could not answer that question: {exc}")

    turns = conversation_turns()
    if turns:
        st.markdown("### Latest answer")
        latest = turns[-1]
        render_answer_card(
            latest["question"],
            latest["result"],
            latest["message_index"],
        )

        previous_turns = turns[:-1]
        if previous_turns:
            with st.expander(f"Previous questions ({len(previous_turns)})"):
                for position, turn in enumerate(reversed(previous_turns)):
                    render_answer_card(
                        turn["question"],
                        turn["result"],
                        turn["message_index"],
                        compact_sources=True,
                    )
                    if position < len(previous_turns) - 1:
                        st.divider()
    elif is_ready:
        st.info("Ask your first question above or choose an example.")

    st.divider()
    st.caption(SAFETY_DISCLAIMER)


if __name__ == "__main__":
    main()
