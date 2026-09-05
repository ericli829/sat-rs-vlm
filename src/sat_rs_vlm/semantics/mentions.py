"""Rule-based object, number, and semantic-fact mention extraction."""

from __future__ import annotations

import re
from typing import Any

from .types import SemanticFacts, TermMention

_ARABIC_NUMBER = re.compile(r"(?<![\w.])-?\d+(?![\w.])")
_ENGLISH_NUMBERS = {
    "no": 0,
    "none": 0,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_CHINESE_NUMBERS = {
    "零": 0,
    "无": 0,
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_SENTENCE_BREAK = re.compile(r"[.!?;。！？；\n]")


def alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    if re.search(r"[a-z0-9]", alias, flags=re.IGNORECASE):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
    return re.compile(escaped, flags=re.IGNORECASE)


def term_mentions(text: str, terms: dict[str, list[str]]) -> list[TermMention]:
    candidates: list[TermMention] = []
    for canonical, aliases in terms.items():
        for alias in aliases:
            for match in alias_pattern(str(alias)).finditer(text):
                candidates.append(
                    TermMention(
                        canonical=str(canonical),
                        alias=str(alias),
                        start=match.start(),
                        end=match.end(),
                    )
                )
    candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.canonical))
    selected: list[TermMention] = []
    for candidate in candidates:
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.start, item.end))


def number_mentions(text: str) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for match in _ARABIC_NUMBER.finditer(text):
        value = int(match.group(0))
        if value >= 0:
            candidates.append((match.start(), match.end(), value))
    for token, value in _ENGLISH_NUMBERS.items():
        for match in alias_pattern(token).finditer(text):
            candidates.append((match.start(), match.end(), value))
    for token, value in _CHINESE_NUMBERS.items():
        for match in re.finditer(re.escape(token), text):
            candidates.append((match.start(), match.end(), value))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if any(
            candidate[0] < existing[1] and existing[0] < candidate[1]
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected


def same_sentence(text: str, left_end: int, right_start: int) -> bool:
    low, high = sorted((left_end, right_start))
    return _SENTENCE_BREAK.search(text[low:high]) is None


def extract_counts(
    text: str,
    objects: list[TermMention],
) -> tuple[tuple[tuple[str, int], ...], list[str]]:
    numbers = number_mentions(text)
    facts: set[tuple[str, int]] = set()
    warnings: list[str] = []
    for start, end, value in numbers:
        candidates: list[tuple[int, int, TermMention]] = []
        for mention in objects:
            if not same_sentence(text, min(mention.end, end), max(mention.start, start)):
                continue
            distance = min(abs(mention.start - end), abs(start - mention.end))
            if distance <= 24:
                direction_penalty = 0 if end <= mention.start else 2
                candidates.append((distance + direction_penalty, mention.start, mention))
        if not candidates:
            continue
        candidates.sort()
        facts.add((candidates[0][2].canonical, value))
    by_object: dict[str, set[int]] = {}
    for object_name, count in facts:
        by_object.setdefault(object_name, set()).add(count)
    for object_name, values in sorted(by_object.items()):
        if len(values) > 1:
            warnings.append(f"conflicting_counts:{object_name}:{sorted(values)}")
    return tuple(sorted(facts)), warnings


def extract_semantic_facts(text: str, ontology: dict[str, Any]) -> SemanticFacts:
    from .relations import extract_changes, extract_relations

    lowered = text.lower()
    objects = term_mentions(lowered, ontology["objects"])
    counts, warnings = extract_counts(lowered, objects)
    relations = extract_relations(lowered, objects, ontology)
    changes = extract_changes(lowered, objects, ontology)
    return SemanticFacts(
        objects=tuple(sorted({mention.canonical for mention in objects})),
        counts=counts,
        relations=relations,
        changes=changes,
        warnings=tuple(warnings),
    )
