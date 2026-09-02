"""Deterministic BM25 plus structural reranking for Planner cheat sheets."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HARD_INTENTS = frozenset(
    {
        "RELATIONAL_COUNT",
        "OBJECT_RELATION",
        "ROUTE_PLANNING",
        "COMPLEX_REASONING",
    }
)

_PHRASES = {
    "left of": "left_of",
    "right of": "right_of",
    "next to": "next_to",
    "in relation to": "in_relation_to",
    "relative to": "relative_to",
    "top to bottom": "top_to_bottom",
    "bottom to top": "bottom_to_top",
}
_RELATION_SIGNALS = (
    "left_of",
    "right_of",
    "next_to",
    "inside",
    "outside",
    "near",
    "between",
    "around",
    "above",
    "below",
    "relative_to",
    "in_relation_to",
)
_COUNT_RE = re.compile(r"\b(?:how many|number of|count|quantity)\b", re.IGNORECASE)
_ROUTE_RE = re.compile(r"\b(?:route|drive|driving|path)\b", re.IGNORECASE)
_COMPLEX_RE = re.compile(
    r"\b(?:why|cause|reason|explain|effect|impact|influence|consequence|likely)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _normalized_text(text: str) -> str:
    value = text.casefold()
    for phrase, replacement in _PHRASES.items():
        value = value.replace(phrase, replacement)
    return value


def lexical_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalized_text(text))


def query_metadata(question: str) -> dict[str, Any]:
    normalized = _normalized_text(question)
    tokens = lexical_tokens(question)
    relations = [signal.upper() for signal in _RELATION_SIGNALS if signal in normalized]
    ordinal = any(
        token in tokens
        for token in ("first", "second", "third", "fourth", "fifth", "leftmost", "rightmost")
    )
    rank = any(token in tokens for token in ("largest", "smallest", "nearest", "farthest"))
    return {
        "relations": relations,
        "relation_depth_proxy": len(relations),
        "count_signal": bool(_COUNT_RE.search(question)),
        "route_signal": bool(_ROUTE_RE.search(question)),
        "ordinal_signal": ordinal,
        "rank_signal": rank,
        "complex_signal": bool(_COMPLEX_RE.search(question)),
    }


def route_hard_intent(question: str) -> str | None:
    metadata = query_metadata(question)
    normalized = _normalized_text(question)
    if metadata["route_signal"]:
        return "ROUTE_PLANNING"
    if metadata["count_signal"] and (
        metadata["relations"]
        or metadata["ordinal_signal"]
        or metadata["rank_signal"]
    ):
        return "RELATIONAL_COUNT"
    if "relative_to" in normalized or "in_relation_to" in normalized:
        return "OBJECT_RELATION"
    if metadata["complex_signal"]:
        return "COMPLEX_REASONING"
    return None


@dataclass(frozen=True)
class RetrievalResult:
    example: dict[str, Any]
    bm25_score: float
    metadata_score: float
    final_score: float
    rank: int

    def log_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example["example_id"],
            "bm25_score": self.bm25_score,
            "metadata_score": self.metadata_score,
            "final_score": self.final_score,
            "rank": self.rank,
        }


class CheatSheetRetriever:
    """Small in-memory BM25 index with deterministic structural reranking."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        ids = [str(example.get("example_id", "")) for example in examples]
        if not examples:
            self.examples: list[dict[str, Any]] = []
            self.documents: list[list[str]] = []
            self.document_frequencies: Counter[str] = Counter()
            self.average_length = 0.0
            return
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("cheat-sheet examples require unique non-empty example_id values")
        non_train = [
            str(example.get("example_id"))
            for example in examples
            if example.get("source_split") != "train"
        ]
        if non_train:
            raise ValueError(f"cheat-sheet index only accepts train examples: {non_train[:10]}")
        self.examples = list(examples)
        self.documents = [lexical_tokens(str(example["question"])) for example in examples]
        self.document_frequencies = Counter(
            token for document in self.documents for token in set(document)
        )
        self.average_length = sum(map(len, self.documents)) / len(self.documents)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> CheatSheetRetriever:
        source = Path(path)
        examples = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(examples)

    def _bm25(self, query: list[str], index: int) -> float:
        if not query or not self.documents:
            return 0.0
        document = self.documents[index]
        frequencies = Counter(document)
        score = 0.0
        k1 = 1.5
        b = 0.75
        population = len(self.documents)
        for token in set(query):
            frequency = frequencies[token]
            if frequency == 0:
                continue
            document_frequency = self.document_frequencies[token]
            inverse = math.log(
                1.0
                + (population - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / max(self.average_length, 1.0)
            )
            score += inverse * frequency * (k1 + 1.0) / denominator
        return score

    @staticmethod
    def _metadata_score(
        query: dict[str, Any], example: dict[str, Any], intent: str | None
    ) -> float:
        metadata = dict(example.get("metadata") or {})
        score = 0.0
        if intent and str(example.get("intent")) == intent:
            score += 4.0
        elif intent:
            score -= 1.0
        query_relations = set(query["relations"])
        example_relations = {str(value) for value in metadata.get("relations") or []}
        score += 1.5 * len(query_relations & example_relations)
        if query["count_signal"] and str(example.get("intent")) == "RELATIONAL_COUNT":
            score += 1.5
        if query["route_signal"] and str(example.get("intent")) == "ROUTE_PLANNING":
            score += 2.0
        if query["ordinal_signal"] and metadata.get("ordinal_signal"):
            score += 1.0
        if query["rank_signal"] and metadata.get("rank_signal"):
            score += 1.0
        query_depth = int(query["relation_depth_proxy"])
        example_depth = int(metadata.get("relation_depth", 0))
        score += max(0.0, 1.0 - 0.25 * abs(query_depth - example_depth))
        return score

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 2,
        intent: str | None = None,
        candidate_k: int = 10,
    ) -> tuple[list[RetrievalResult], float]:
        started = time.perf_counter()
        if top_k <= 0 or not self.examples:
            return [], (time.perf_counter() - started) * 1000.0
        tokens = lexical_tokens(question)
        bm25 = [self._bm25(tokens, index) for index in range(len(self.examples))]
        lexical_candidates = sorted(
            range(len(self.examples)),
            key=lambda index: (-bm25[index], str(self.examples[index]["example_id"])),
        )[: max(top_k, candidate_k)]
        query = query_metadata(question)
        scored = []
        for index in lexical_candidates:
            metadata_score = self._metadata_score(query, self.examples[index], intent)
            scored.append((index, bm25[index], metadata_score, bm25[index] + metadata_score))
        scored.sort(
            key=lambda item: (-item[3], -item[1], str(self.examples[item[0]]["example_id"]))
        )
        results = [
            RetrievalResult(
                example=self.examples[index],
                bm25_score=bm25_score,
                metadata_score=metadata_score,
                final_score=final_score,
                rank=rank,
            )
            for rank, (index, bm25_score, metadata_score, final_score) in enumerate(
                scored[:top_k], 1
            )
        ]
        return results, (time.perf_counter() - started) * 1000.0


def compose_cheat_sheet_prompt(
    base_instruction: str,
    *,
    rule_cards: str = "",
    retrieved: list[RetrievalResult] | None = None,
) -> str:
    sections = [base_instruction.rstrip()]
    if rule_cards.strip():
        sections.append("[CHEAT SHEET RULES]\n" + rule_cards.strip())
    for index, result in enumerate(retrieved or [], 1):
        example = result.example
        sections.append(
            f"[RETRIEVED EXAMPLE {index}]\n"
            f"Question:\n{example['question']}\n\n"
            f"Plan:\n{example['dsl']}"
        )
    if len(sections) > 1:
        sections.append(
            "Examples illustrate planning patterns. Compile the current question "
            "independently. Do not copy object names, node counts, relations, or "
            "references unless the current question requires them. Return DSL only."
        )
    return "\n\n".join(sections)
