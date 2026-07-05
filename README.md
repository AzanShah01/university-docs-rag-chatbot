# University Docs RAG Chatbot

A private, zero-cost retrieval-augmented generation (RAG) application for asking questions about university documents. Upload text-based PDFs such as thesis guidelines, course outlines, regulations, timetables, or academic policies; the app retrieves the most relevant passages and creates an extractive answer using only text found in those passages.

No API key, paid service, cloud platform, LangChain, or hosted language model is required. Document processing happens locally. The app binds to `localhost` by default. The first use downloads the embedding model; once cached, the workflow can run without an internet connection.

## Overview

University information is often scattered across long PDF files. This project provides a focused search and question-answering interface that preserves the source document, page number, chunk identifier, and similarity score for every retrieved passage.

Unlike a generative chatbot, this first version uses extractive question answering. It selects relevant sentences from the uploaded documents rather than generating unsupported text.

## Problem statement

Students and staff can spend significant time searching policy documents for a specific deadline, requirement, rule, or procedure. Ordinary keyword search may miss related wording, while general-purpose chatbots may invent details or rely on information outside the document. This application combines semantic retrieval with source-grounded extraction to make university PDFs easier to navigate.

## Features

- Upload and process one or multiple PDF documents.
- Extract text locally with PyMuPDF.
- Preserve document names and page numbers.
- Split pages into overlapping 400-word chunks with 60-word context overlap.
- Create lightweight embeddings with `all-MiniLM-L6-v2`.
- Search a normalized FAISS index using cosine similarity.
- Produce an extractive answer from relevant source sentences only.
- Display source chunks, page numbers, chunk IDs, and similarity scores.
- Download the answer and source references as a text file.
- Keep a local chat history with example questions and clear/reset controls.
- Label answers as high, medium, or low confidence and flag weak matches.
- Handle empty, scanned, protected, unreadable, and invalid PDFs gracefully.
- Cache the embedding model to reduce repeat loading time.

## Tech stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS CPU
- NumPy
- Pandas

## Project structure

```text
university-docs-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── sample_docs/
│   └── README.md
├── screenshots/
│   └── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── pdf_loader.py
    ├── text_splitter.py
    ├── vector_store.py
    ├── rag_pipeline.py
    └── utils.py
```

## How it works

1. The user uploads one or more PDF files.
2. PyMuPDF extracts text from every readable page and records its source metadata.
3. Text is normalized and divided into overlapping word chunks.
4. Sentence Transformers converts each chunk into a local vector embedding.
5. FAISS stores normalized vectors and retrieves the closest chunks to a question.
6. Relevant sentences are ranked semantically and copied into a concise extractive answer.
7. The answer is shown alongside the source chunks used during retrieval.

## Installation

Python 3.10–3.12 is recommended for broad package compatibility.

```powershell
git clone <your-repository-url>
cd university-docs-rag-chatbot
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first time you process documents, the app downloads the free `all-MiniLM-L6-v2` model to the local Sentence Transformers cache. Later runs try that local cache first, so PDF extraction, chunking, embeddings, FAISS search, and extractive answers all work without an internet connection. Uploaded PDFs stay in the current Streamlit process and are not sent to an AI API.

## How to run

From the project root, run:

```powershell
streamlit run app.py
```

Open the local address displayed by Streamlit, upload a PDF, select **Process documents**, and enter a focused question.

The page appears before the machine-learning model loads. Model loading begins only after you select **Process documents**, and the first run can take longer while the model downloads.

## Example use cases

- Ask which documents are required for thesis submission.
- Find attendance or grading rules in a course outline.
- Check an academic calendar or examination timetable.
- Locate eligibility conditions in scholarship regulations.
- Review administrative procedures in a student handbook.

## Screenshots

Add screenshots of the running application to the `screenshots/` directory before publishing the project. Suggested screenshots include the upload screen, processed-document summary, answer, and expanded source evidence.

## Limitations

- Image-only and scanned PDFs require OCR, which is not included in this version.
- Extractive answers may sound less conversational than generative answers.
- Multi-column layouts, tables, formulas, or unusual PDF encoding may reduce extraction quality.
- Retrieval quality depends on document wording and the specificity of the question.
- The first model download requires an internet connection and local disk space.
- Uploaded documents and the FAISS index remain in application memory and reset when the session ends.

## Future improvements

- Add optional local OCR for scanned PDFs.
- Add table-aware extraction and richer citations.
- Support persistent local indexes for frequently used public documents.
- Add hybrid semantic and keyword retrieval.
- Add reranking and configurable chunk settings.
- Offer an optional fully local generative model for capable hardware.
- Add automated tests and PDF fixtures.

## Safety disclaimer

This tool answers only from uploaded documents and may not be 100% accurate. Always verify important academic or administrative information from official university sources.

Do not upload private, confidential, or sensitive documents to shared computers. This tool is an information aid and is not an official university authority.

## Author

Azan Shah
