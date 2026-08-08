"""标准库实现的遥感对象、计数、空间关系和变化事件抽取器。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TermMention:
    canonical: str
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class SemanticFacts:
    objects: tuple[str, ...]
    counts: tuple[tuple[str, int], ...]
    relations: tuple[tuple[str, str, str], ...]
    changes: tuple[tuple[str | None, str], ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": list(self.objects),
            "counts": [
                {"object": object_name, "count": count} for object_name, count in self.counts
            ],
            "relations": [
                {"subject": subject, "predicate": predicate, "object": object_name}
                for subject, predicate, object_name in self.relations
            ],
            "changes": [
                {"object": object_name, "change_type": change_type}
                for object_name, change_type in self.changes
            ],
            "warnings": list(self.warnings),
        }


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


def load_ontology(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"semantic ontology does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid semantic ontology JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("semantic ontology must be a JSON object")
    for field in ("ontology_version", "objects", "relations", "changes"):
        if field not in payload:
            raise ValueError(f"semantic ontology is missing {field}")
    if not all(isinstance(payload[field], dict) for field in ("objects", "relations", "changes")):
        raise ValueError("objects, relations and changes must be JSON objects")
    return payload


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    if re.search(r"[a-z0-9]", alias, flags=re.IGNORECASE):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
    return re.compile(escaped, flags=re.IGNORECASE)


def _term_mentions(text: str, terms: dict[str, list[str]]) -> list[TermMention]:
    candidates: list[TermMention] = []
    for canonical, aliases in terms.items():
        for alias in aliases:
            for match in _alias_pattern(str(alias)).finditer(text):
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
        overlaps = any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        )
        if overlaps:
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.start, item.end))


def _number_mentions(text: str) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for match in _ARABIC_NUMBER.finditer(text):
        value = int(match.group(0))
        if value >= 0:
            candidates.append((match.start(), match.end(), value))
    for token, value in _ENGLISH_NUMBERS.items():
        for match in _alias_pattern(token).finditer(text):
            candidates.append((match.start(), match.end(), value))
    for token, value in _CHINESE_NUMBERS.items():
        for match in re.finditer(re.escape(token), text):
            candidates.append((match.start(), match.end(), value))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if any(candidate[0] < existing[1] and existing[0] < candidate[1] for existing in selected):
            continue
        selected.append(candidate)
    return selected


def _same_sentence(text: str, left_end: int, right_start: int) -> bool:
    low, high = sorted((left_end, right_start))
    return _SENTENCE_BREAK.search(text[low:high]) is None


def _extract_counts(
    text: str,
    objects: list[TermMention],
) -> tuple[tuple[tuple[str, int], ...], list[str]]:
    numbers = _number_mentions(text)
    facts: set[tuple[str, int]] = set()
    warnings: list[str] = []
    for start, end, value in numbers:
        candidates: list[tuple[int, int, TermMention]] = []
        for mention in objects:
            if not _same_sentence(text, min(mention.end, end), max(mention.start, start)):
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


def _relation_mentions(text: str, ontology: dict[str, Any]) -> list[TermMention]:
    terms = {
        canonical: [str(alias) for alias in spec.get("aliases", [])]
        for canonical, spec in ontology["relations"].items()
        if isinstance(spec, dict)
    }
    return _term_mentions(text, terms)


def _nearest_object(
    text: str,
    relation: TermMention,
    objects: list[TermMention],
    *,
    before: bool,
) -> TermMention | None:
    candidates: list[tuple[int, TermMention]] = []
    for mention in objects:
        if before and mention.end <= relation.start:
            distance = relation.start - mention.end
        elif not before and mention.start >= relation.end:
            distance = mention.start - relation.end
        else:
            continue
        same_sentence = (
            _same_sentence(text, mention.end, relation.start)
            if before
            else _same_sentence(text, relation.end, mention.start)
        )
        if distance <= 80 and same_sentence:
            candidates.append((distance, mention))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _extract_relations(
    text: str,
    objects: list[TermMention],
    ontology: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    facts: set[tuple[str, str, str]] = set()
    for relation in _relation_mentions(text, ontology):
        subject = _nearest_object(text, relation, objects, before=True)
        object_mention = _nearest_object(text, relation, objects, before=False)
        if (
            subject is None
            or object_mention is None
            or subject.canonical == object_mention.canonical
        ):
            continue
        left, right = subject.canonical, object_mention.canonical
        spec = ontology["relations"].get(relation.canonical, {})
        if isinstance(spec, dict) and spec.get("symmetric"):
            left, right = sorted((left, right))
        facts.add((left, relation.canonical, right))
    return tuple(sorted(facts))


def _extract_changes(
    text: str,
    objects: list[TermMention],
    ontology: dict[str, Any],
) -> tuple[tuple[str | None, str], ...]:
    change_mentions = _term_mentions(text, ontology["changes"])
    facts: set[tuple[str | None, str]] = set()
    for change in change_mentions:
        candidates: list[tuple[int, TermMention]] = []
        for mention in objects:
            distance = min(abs(change.start - mention.end), abs(mention.start - change.end))
            same_sentence = _same_sentence(
                text,
                min(change.end, mention.end),
                max(change.start, mention.start),
            )
            if distance <= 80 and same_sentence:
                candidates.append((distance, mention))
        object_name = min(candidates, key=lambda item: item[0])[1].canonical if candidates else None
        facts.add((object_name, change.canonical))
    return tuple(sorted(facts, key=lambda item: (item[0] or "", item[1])))


def extract_semantic_facts(text: str, ontology: dict[str, Any]) -> SemanticFacts:
    lowered = text.lower()
    objects = _term_mentions(lowered, ontology["objects"])
    counts, warnings = _extract_counts(lowered, objects)
    relations = _extract_relations(lowered, objects, ontology)
    changes = _extract_changes(lowered, objects, ontology)
    return SemanticFacts(
        objects=tuple(sorted({mention.canonical for mention in objects})),
        counts=counts,
        relations=relations,
        changes=changes,
        warnings=tuple(warnings),
    )
