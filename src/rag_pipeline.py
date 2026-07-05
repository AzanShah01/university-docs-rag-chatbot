"""Retrieval and local extractive question-answering pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH_SIZE, MIN_SIMILARITY_SCORE, TOP_K
from src.vector_store import FAISSVectorStore


NO_ANSWER = "I could not find a clear answer in the uploaded documents."
STOP_WORDS = {
    "a", "an", "and", "are", "can", "did", "do", "does", "for", "from",
    "how", "in", "is", "of", "on", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "why",
}
STATISTICS_WORDS = {"count", "columns", "number", "records", "statistic", "statistics", "total"}
INCOMPLETE_ENDINGS = {"and", "as", "because", "for", "from", "including", "of", "or", "such", "the", "to", "using", "with"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def looks_like_heading(text: str) -> bool:
    """Return whether a short PDF line looks like a section heading."""
    tokens = text.split()
    if not tokens or len(tokens) > 12 or text.rstrip().endswith((".", "!", "?", ":")):
        return False
    numbered = bool(re.match(r"^\d+(?:\.\d+)*\s+", text))
    title_case_ratio = sum(token[:1].isupper() for token in tokens) / len(tokens)
    return numbered or title_case_ratio >= 0.65


def split_units(text: str) -> list[str]:
    """Create complete sentence candidates while retaining section headings."""
    units: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            units.extend(
                clean
                for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", " ".join(buffer))
                if (clean := re.sub(r"\s+", " ", part).strip(" -\t"))
            )
            buffer.clear()

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" -\t")
        if not line:
            flush_buffer()
        elif looks_like_heading(line):
            flush_buffer()
            units.append(line)
        else:
            buffer.append(line)
            if line.endswith((".", "!", "?")):
                flush_buffer()
    flush_buffer()
    return units


def split_sentences(text: str) -> list[str]:
    """Split source text into readable answer candidates without rewriting it."""
    return [unit for unit in split_units(text) if len(unit.split()) >= 4]


def is_table_fragment(text: str) -> bool:
    """Identify flattened table headers/statistics that make poor prose answers."""
    tokens = text.split()
    if not tokens:
        return True
    numeric_count = sum(bool(re.search(r"\d", token)) for token in tokens)
    capitalized_count = sum(token[:1].isupper() for token in tokens)
    no_sentence_punctuation = not text.rstrip().endswith((".", "!", "?"))
    return (
        numeric_count >= 3
        or (len(tokens) >= 8 and no_sentence_punctuation
            and capitalized_count / len(tokens) >= 0.45)
    )


def is_incomplete_sentence(text: str) -> bool:
    """Reject headings, clipped lines, and dangling sentence fragments."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < 5 or looks_like_heading(text):
        return True
    return tokens[-1] in INCOMPLETE_ENDINGS


class RAGPipeline:
    """Coordinate local embeddings, FAISS retrieval, and extractive answers."""

    def __init__(self, model: SentenceTransformer | Any) -> None:
        self.model = model
        self.vector_store = FAISSVectorStore()
        self.chunks: list[dict] = []

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=EMBEDDING_BATCH_SIZE,
            ),
            dtype="float32",
        )

    def index_chunks(self, chunks: list[dict]) -> None:
        embeddings = self._encode([chunk["text"] for chunk in chunks])
        self.vector_store.add(embeddings, chunks)
        self.chunks = list(chunks)

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready

    @property
    def chunk_count(self) -> int:
        return len(self.vector_store)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        query_embedding = self._encode([question])
        return self.vector_store.search(query_embedding, top_k=top_k)

    def _extract_answer(
        self,
        question: str,
        sources: list[dict],
        min_similarity: float,
    ) -> tuple[str, float]:
        """Return the strongest non-duplicate source sentences and their score."""
        if not sources or sources[0]["score"] < min_similarity:
            return NO_ANSWER, 0.0

        question_words = _words(question) - STOP_WORDS
        wants_tabular_answer = bool(question_words & STATISTICS_WORDS)
        candidates: list[tuple[str, float, float]] = []
        for source in sources:
            units = split_units(source["text"])[:30]
            anchor_positions = []
            for position, unit in enumerate(units):
                unit_words = _words(unit)
                coverage = len(question_words & unit_words) / max(1, len(question_words))
                if question.lower() in unit.lower() or coverage >= 0.75:
                    anchor_positions.append(position)

            for position, sentence in enumerate(units):
                if is_incomplete_sentence(sentence):
                    continue
                if is_table_fragment(sentence) and not wants_tabular_answer:
                    continue
                distance = min(
                    (abs(position - anchor) for anchor in anchor_positions),
                    default=4,
                )
                heading_proximity = max(0.0, 1.0 - (distance / 4.0))
                candidates.append(
                    (sentence, max(0.0, source["score"]), heading_proximity)
                )

        if not candidates:
            return NO_ANSWER, 0.0

        question_embedding = self._encode([question])[0]
        sentence_embeddings = self._encode([item[0] for item in candidates])
        semantic_scores = sentence_embeddings @ question_embedding

        ranked: list[tuple[float, str]] = []
        for index, (sentence, chunk_score, heading_proximity) in enumerate(candidates):
            sentence_words = _words(sentence)
            lexical_score = len(question_words & sentence_words) / max(1, len(question_words))
            phrase_bonus = 0.08 if question.lower() in sentence.lower() else 0.0
            table_penalty = 0.20 if is_table_fragment(sentence) else 0.0
            combined_score = (
                (0.45 * float(semantic_scores[index]))
                + (0.30 * chunk_score)
                + (0.20 * lexical_score)
                + (0.10 * heading_proximity)
                + phrase_bonus
                - table_penalty
            )
            ranked.append((combined_score, sentence))
        ranked.sort(key=lambda item: item[0], reverse=True)

        if not ranked or ranked[0][0] < min_similarity:
            return NO_ANSWER, ranked[0][0] if ranked else 0.0

        selected: list[str] = []
        seen_words: list[set[str]] = []
        for score, sentence in ranked:
            if score < min_similarity or len(selected) >= 3:
                break
            words = _words(sentence)
            is_duplicate = any(
                len(words & previous) / max(1, len(words | previous)) > 0.60
                for previous in seen_words
            )
            if not is_duplicate:
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith((".", "!", "?")):
                    clean_sentence += "."
                selected.append(clean_sentence)
                seen_words.append(words)

        if not selected:
            return NO_ANSWER, ranked[0][0]
        return " ".join(selected), ranked[0][0]

    @staticmethod
    def _confidence_label(
        top_similarity: float,
        answer_score: float,
        has_answer: bool,
    ) -> str:
        if not has_answer:
            return "Low confidence"
        if top_similarity >= 0.55:
            return "High confidence"
        if top_similarity >= 0.35:
            return "Medium confidence"
        return "Low confidence"

    def answer_question(
        self,
        question: str,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY_SCORE,
    ) -> dict:
        clean_question = question.strip()
        if not clean_question:
            return {
                "question": question,
                "answer": NO_ANSWER,
                "sources": [],
                "confidence": "Low confidence",
                "confidence_score": 0.0,
                "guidance": "",
                "answer_found": False,
            }

        retrieved = self.retrieve(clean_question, top_k=top_k)
        top_similarity = retrieved[0]["score"] if retrieved else 0.0
        relevant_sources = [
            source for source in retrieved if source["score"] >= min_similarity
        ]
        answer, answer_score = self._extract_answer(
            clean_question,
            relevant_sources,
            min_similarity,
        )
        has_answer = answer != NO_ANSWER
        confidence = self._confidence_label(top_similarity, answer_score, has_answer)
        sources = (relevant_sources if relevant_sources else retrieved)[:top_k]
        return {
            "question": clean_question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "confidence_score": top_similarity,
            "answer_found": has_answer,
            "guidance": (
                "This answer is based on weak document matches. Please verify from the source evidence."
                if confidence == "Low confidence" and has_answer
                else ""
            ),
        }
