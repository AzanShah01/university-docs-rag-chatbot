"""Retrieval and local extractive question-answering pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_BATCH_SIZE,
    MIN_KEYWORD_OVERLAP,
    MIN_SIMILARITY_SCORE,
    SECTION_QUERY_MAX_WORDS,
    TOP_K,
)
from src.vector_store import FAISSVectorStore


NO_ANSWER = "I could not find a clear answer in the uploaded documents."
STRUCTURE_ONLY_ANSWER = (
    "I found this topic in the document structure, but not enough explanatory "
    "content was retrieved. Try asking a more specific question or check the section page."
)
STOP_WORDS = {
    "a", "an", "and", "are", "can", "did", "do", "does", "for", "from",
    "how", "in", "is", "of", "on", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "why",
}
STATISTICS_WORDS = {"count", "columns", "number", "records", "statistic", "statistics", "total"}
INCOMPLETE_ENDINGS = {"and", "as", "because", "for", "from", "including", "of", "or", "such", "the", "to", "using", "with"}
SECTION_PHRASES = {
    "introduction",
    "literature review",
    "future work",
    "system architecture",
    "dataset collection",
    "methodology",
    "limitations",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def meaningful_words(text: str) -> set[str]:
    return _words(text) - STOP_WORDS


def normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def is_section_query(question: str, question_words: set[str]) -> bool:
    normalized = normalized_phrase(question)
    return (
        len(question_words) <= SECTION_QUERY_MAX_WORDS
        and any(phrase in normalized for phrase in SECTION_PHRASES)
    )


def phrase_matches(question: str, text: str) -> bool:
    question_normalized = normalized_phrase(question)
    text_normalized = normalized_phrase(text)
    meaningful_phrase = " ".join(
        word for word in question_normalized.split() if word not in STOP_WORDS
    )
    if len(meaningful_phrase.split()) >= 2 and meaningful_phrase in text_normalized:
        return True
    return any(
        phrase in question_normalized and phrase in text_normalized
        for phrase in SECTION_PHRASES
    )


def chunk_match_metadata(question: str, chunk: dict) -> dict:
    """Compute lexical, phrase, section, and low-value reranking signals."""
    question_words = meaningful_words(question)
    text = chunk.get("text", "")
    section_title = chunk.get("section_title", "")
    chunk_words = meaningful_words(text)
    section_words = meaningful_words(section_title)
    keyword_overlap = len(question_words & chunk_words)
    heading_overlap = len(question_words & section_words) / max(1, len(question_words))
    exact_phrase = phrase_matches(question, text)
    section_phrase = phrase_matches(question, section_title)
    section_match = min(1.0, heading_overlap + (0.35 if section_phrase else 0.0))
    heading_weight = 0.40 if is_section_query(question, question_words) else 0.25
    keyword_score = min(
        1.0,
        keyword_overlap / max(1, min(len(question_words), 4)),
    )
    similarity = max(0.0, float(chunk.get("score", 0.0)))
    combined_score = (
        (0.55 * similarity)
        + (0.18 * keyword_score)
        + (0.12 if exact_phrase else 0.0)
        + (heading_weight * section_match)
        - (0.65 if chunk.get("low_value", False) else 0.0)
    )
    return {
        "keyword_overlap": keyword_overlap,
        "exact_phrase_match": exact_phrase,
        "section_match": section_match,
        "combined_score": max(-1.0, min(1.0, combined_score)),
    }


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
        """Retrieve broadly with FAISS, then rerank using document structure."""
        query_embedding = self._encode([question])
        candidate_count = min(len(self.chunks), max(30, top_k * 6))
        semantic_candidates = self.vector_store.search(
            query_embedding,
            top_k=candidate_count,
        )
        candidates_by_id = {
            candidate["chunk_id"]: candidate for candidate in semantic_candidates
        }

        # Add strong lexical/heading matches even if FAISS placed them below its window.
        for chunk in self.chunks:
            if chunk["chunk_id"] in candidates_by_id:
                continue
            lexical_candidate = dict(chunk)
            lexical_candidate["score"] = 0.0
            metadata = chunk_match_metadata(question, lexical_candidate)
            if (
                metadata["keyword_overlap"] >= MIN_KEYWORD_OVERLAP
                or metadata["exact_phrase_match"]
                or metadata["section_match"] >= 0.35
            ):
                candidates_by_id[chunk["chunk_id"]] = lexical_candidate

        reranked = []
        for candidate in candidates_by_id.values():
            ranked_candidate = dict(candidate)
            ranked_candidate.update(chunk_match_metadata(question, ranked_candidate))
            reranked.append(ranked_candidate)
        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        selected = reranked[:top_k]
        structural_match = next(
            (
                item for item in reranked
                if item.get("low_value", False)
                and (
                    item.get("keyword_overlap", 0) >= MIN_KEYWORD_OVERLAP
                    or item.get("exact_phrase_match", False)
                    or item.get("section_match", 0.0) >= 0.35
                )
            ),
            None,
        )
        if structural_match and structural_match not in selected:
            if len(selected) >= top_k:
                selected[-1] = structural_match
            else:
                selected.append(structural_match)
            selected.sort(key=lambda item: item["combined_score"], reverse=True)
        return selected

    @staticmethod
    def _is_meaningful_match(
        question: str,
        source: dict,
        min_similarity: float,
    ) -> bool:
        structural_signal = bool(
            source.get("keyword_overlap", 0) >= MIN_KEYWORD_OVERLAP
            or source.get("exact_phrase_match", False)
            or source.get("section_match", 0.0) >= 0.35
        )
        if structural_signal:
            return True
        semantic_threshold = (
            max(0.45, min_similarity)
            if is_section_query(question, meaningful_words(question))
            else min_similarity
        )
        return source.get("score", 0.0) >= semantic_threshold

    def _extract_answer(
        self,
        question: str,
        sources: list[dict],
        min_similarity: float,
    ) -> tuple[str, float]:
        """Return the strongest non-duplicate source sentences and their score."""
        if not sources:
            return NO_ANSWER, 0.0

        question_words = meaningful_words(question)
        wants_tabular_answer = bool(question_words & STATISTICS_WORDS)
        candidates: list[tuple[str, float, float]] = []
        for source in sources:
            units = split_units(source["text"])[:40]
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
                heading_proximity = max(
                    0.0,
                    1.0 - (distance / 4.0),
                    0.75 * source.get("section_match", 0.0),
                )
                candidates.append(
                    (
                        sentence,
                        max(0.0, source.get("combined_score", 0.0)),
                        heading_proximity,
                    )
                )

        if not candidates:
            return NO_ANSWER, 0.0

        question_embedding = self._encode([question])[0]
        sentence_embeddings = self._encode([item[0] for item in candidates])
        semantic_scores = sentence_embeddings @ question_embedding

        ranked: list[tuple[float, str]] = []
        for index, (sentence, chunk_score, heading_proximity) in enumerate(candidates):
            sentence_words = meaningful_words(sentence)
            lexical_score = len(question_words & sentence_words) / max(1, len(question_words))
            phrase_bonus = 0.08 if phrase_matches(question, sentence) else 0.0
            table_penalty = 0.20 if is_table_fragment(sentence) else 0.0
            combined_score = (
                (0.40 * float(semantic_scores[index]))
                + (0.30 * chunk_score)
                + (0.20 * lexical_score)
                + (0.15 * heading_proximity)
                + phrase_bonus
                - table_penalty
            )
            ranked.append((combined_score, sentence))
        ranked.sort(key=lambda item: item[0], reverse=True)

        sentence_threshold = 0.18
        if not ranked or ranked[0][0] < sentence_threshold:
            return NO_ANSWER, ranked[0][0] if ranked else 0.0

        selected: list[str] = []
        seen_words: list[set[str]] = []
        for score, sentence in ranked:
            if score < sentence_threshold or len(selected) >= 5:
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
                "structure_only": False,
            }

        retrieved = self.retrieve(clean_question, top_k=top_k)
        meaningful_sources = [
            source
            for source in retrieved
            if self._is_meaningful_match(clean_question, source, min_similarity)
        ]
        useful_sources = [
            source for source in meaningful_sources if not source.get("low_value", False)
        ]
        structural_sources = [
            source for source in meaningful_sources if source.get("low_value", False)
        ]

        structure_only = False
        if useful_sources:
            answer, answer_score = self._extract_answer(
                clean_question,
                useful_sources,
                min_similarity,
            )
            sources = useful_sources[:top_k]
        elif structural_sources:
            answer = STRUCTURE_ONLY_ANSWER
            answer_score = structural_sources[0].get("combined_score", 0.0)
            sources = structural_sources[:top_k]
            structure_only = True
        else:
            answer = NO_ANSWER
            answer_score = 0.0
            sources = []

        has_answer = answer not in {NO_ANSWER, STRUCTURE_ONLY_ANSWER}
        confidence_score = (
            sources[0].get("combined_score", 0.0) if sources else 0.0
        )
        confidence = self._confidence_label(
            confidence_score,
            answer_score,
            has_answer,
        )
        return {
            "question": clean_question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "answer_found": has_answer,
            "structure_only": structure_only,
            "guidance": (
                "This answer is based on weak document matches. Please verify from the source evidence."
                if confidence == "Low confidence" and has_answer
                else ""
            ),
        }
