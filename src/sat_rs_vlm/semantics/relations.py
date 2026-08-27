"""Relation and change-event extraction for the common semantic layer."""

from __future__ import annotations

from typing import Any

from .mentions import same_sentence, term_mentions
from .types import TermMention


def relation_mentions(text: str, ontology: dict[str, Any]) -> list[TermMention]:
    terms = {
        canonical: [str(alias) for alias in spec.get("aliases", [])]
        for canonical, spec in ontology["relations"].items()
        if isinstance(spec, dict)
    }
    return term_mentions(text, terms)


def nearest_object(
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
        same = (
            same_sentence(text, mention.end, relation.start)
            if before
            else same_sentence(text, relation.end, mention.start)
        )
        if distance <= 80 and same:
            candidates.append((distance, mention))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def extract_relations(
    text: str,
    objects: list[TermMention],
    ontology: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    facts: set[tuple[str, str, str]] = set()
    for relation in relation_mentions(text, ontology):
        subject = nearest_object(text, relation, objects, before=True)
        object_mention = nearest_object(text, relation, objects, before=False)
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


def extract_changes(
    text: str,
    objects: list[TermMention],
    ontology: dict[str, Any],
) -> tuple[tuple[str | None, str], ...]:
    change_mentions = term_mentions(text, ontology["changes"])
    facts: set[tuple[str | None, str]] = set()
    for change in change_mentions:
        candidates: list[tuple[int, TermMention]] = []
        for mention in objects:
            distance = min(abs(change.start - mention.end), abs(mention.start - change.end))
            same = same_sentence(
                text,
                min(change.end, mention.end),
                max(change.start, mention.start),
            )
            if distance <= 80 and same:
                candidates.append((distance, mention))
        object_name = (
            min(candidates, key=lambda item: item[0])[1].canonical if candidates else None
        )
        facts.add((object_name, change.canonical))
    return tuple(sorted(facts, key=lambda item: (item[0] or "", item[1])))
