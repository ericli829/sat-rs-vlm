"""Image-grounded LEVIR-CC change-semantic evaluation helpers.

Gold labels are created by annotators inspecting the before/after image pair.
Model captions are converted to a fixed, auditable event schema before
comparison.  This module deliberately does not treat the reference caption as
visual ground truth.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

OBJECT_LABELS = frozenset(
    {
        "building",
        "road",
        "parking_area",
        "bridge",
        "sports_field",
        "water_body",
        "vegetation_landcover",
        "other_permanent_structure",
    }
)
DIRECTION_LABELS = frozenset(
    {
        "appearance_construction",
        "disappearance_demolition",
        "expansion",
        "reduction",
        "replacement_modification",
        "state_change_unspecified",
    }
)
OPPOSITE_DIRECTIONS = {
    "appearance_construction": "disappearance_demolition",
    "disappearance_demolition": "appearance_construction",
    "expansion": "reduction",
    "reduction": "expansion",
}

_OBJECT_ALIASES = {
    "building": (
        "building",
        "buildings",
        "house",
        "houses",
        "home",
        "homes",
        "villa",
        "villas",
        "structure",
        "structures",
        "roof",
        "roofs",
    ),
    "road": ("road", "roads", "street", "streets", "highway", "highways", "path", "paths"),
    "parking_area": ("parking lot", "parking lots", "parking area", "parking areas"),
    "bridge": ("bridge", "bridges"),
    "sports_field": ("sports field", "sports fields", "playground", "playgrounds", "stadium"),
    "water_body": ("lake", "lakes", "pond", "ponds", "river", "rivers", "water body"),
    "vegetation_landcover": (
        "forest",
        "forests",
        "vegetation",
        "trees",
        "tree",
        "grass",
        "farmland",
        "field",
        "fields",
        "bareland",
    ),
    "other_permanent_structure": (
        "walkway",
        "walkways",
        "driveway",
        "driveways",
        "runway",
        "runways",
        "facility",
        "facilities",
    ),
}
_DIRECTION_ALIASES = {
    "appearance_construction": (
        "new",
        "newly",
        "appeared",
        "appears",
        "emerged",
        "built",
        "constructed",
        "added",
        "addition",
    ),
    "disappearance_demolition": (
        "disappeared",
        "removed",
        "demolished",
        "destroyed",
        "torn down",
        "no longer present",
        "gone",
    ),
    "expansion": ("expanded", "enlarged", "extended", "widened", "grew larger"),
    "reduction": ("reduced", "shrank", "contracted", "narrowed", "became smaller"),
    "replacement_modification": (
        "replaced",
        "converted",
        "transformed",
        "reconfigured",
        "modified",
    ),
    "state_change_unspecified": ("changed", "change", "different", "altered"),
}
_NO_CHANGE = re.compile(
    r"\b(?:no\s+(?:meaningful\s+)?change|unchanged|remain(?:s|ed)?\s+the\s+same|"
    r"no\s+difference|identical)\b",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[.!?;]")


@dataclass(frozen=True)
class VisualSemanticFacts:
    """Structured change claims extracted from one generated Caption."""

    # ``None`` means the Caption did not express a resolvable binary decision;
    # it is not silently credited as a no-change prediction.
    change_label: int | None
    objects: frozenset[str]
    directions: frozenset[str]
    events: frozenset[tuple[str, str]]
    extraction_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_label": self.change_label,
            "objects": sorted(self.objects),
            "directions": sorted(self.directions),
            "events": [
                {"object": object_name, "direction": direction}
                for object_name, direction in sorted(self.events)
            ],
            "extraction_notes": list(self.extraction_notes),
        }


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


def _mentions(text: str, aliases: dict[str, tuple[str, ...]]) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    for label, values in aliases.items():
        for alias in values:
            for match in _alias_pattern(alias).finditer(text):
                found.append((label, match.start(), match.end()))
    found.sort(key=lambda item: (item[1], -(item[2] - item[1]), item[0]))
    selected: list[tuple[str, int, int]] = []
    for candidate in found:
        if any(candidate[1] < item[2] and item[1] < candidate[2] for item in selected):
            continue
        selected.append(candidate)
    return selected


def _same_sentence(text: str, left: int, right: int) -> bool:
    low, high = sorted((left, right))
    return _SENTENCE_BREAK.search(text[low:high]) is None


def _nearest_object(
    text: str, direction: tuple[str, int, int], objects: list[tuple[str, int, int]]
) -> str | None:
    candidates: list[tuple[int, str]] = []
    _, start, end = direction
    for object_name, object_start, object_end in objects:
        if not _same_sentence(text, min(start, object_start), max(end, object_end)):
            continue
        distance = min(abs(start - object_end), abs(object_start - end))
        if distance <= 96:
            candidates.append((distance, object_name))
    return min(candidates)[1] if candidates else None


def _direction_is_negated(text: str, direction: tuple[str, int, int]) -> bool:
    """Reject claims such as ``no building was demolished``.

    This is deliberately a narrow output-postprocessing rule. It handles the
    common local negation form without attempting unrestricted natural-language
    inference.
    """

    _, start, _end = direction
    prefix = text[max(0, start - 40) : start]
    return bool(re.search(r"\b(?:no|not|without|never)\b[^.;!?]{0,32}$", prefix))


def _stative_appearance(text: str, direction: tuple[str, int, int]) -> bool:
    """Do not treat ``appears to be unchanged`` as temporal appearance."""

    name, _start, end = direction
    return name == "appearance_construction" and bool(re.match(r"\s+to\b", text[end:]))


def extract_visual_semantics(caption: str) -> VisualSemanticFacts:
    """Extract a fixed event schema from a generated English LEVIR Caption."""

    text = caption.strip().lower()
    if not text:
        return VisualSemanticFacts(None, frozenset(), frozenset(), frozenset(), ("empty_caption",))
    objects = _mentions(text, _OBJECT_ALIASES)
    directions = [
        direction
        for direction in _mentions(text, _DIRECTION_ALIASES)
        if not _direction_is_negated(text, direction) and not _stative_appearance(text, direction)
    ]
    events: set[tuple[str, str]] = set()
    notes: list[str] = []
    for direction in directions:
        direction_name = direction[0]
        object_name = _nearest_object(text, direction, objects)
        if object_name is None:
            notes.append(f"direction_without_object:{direction_name}")
            continue
        events.add((object_name, direction_name))
    if not events and not directions and _NO_CHANGE.search(text):
        return VisualSemanticFacts(
            0, frozenset(), frozenset(), frozenset(), tuple(sorted(set(notes)))
        )
    if not events:
        if directions:
            direction_names = frozenset(direction[0] for direction in directions)
            return VisualSemanticFacts(
                1,
                frozenset(),
                direction_names,
                frozenset(),
                tuple(sorted(set(notes + ["change_without_resolved_object"]))),
            )
        return VisualSemanticFacts(
            None,
            frozenset(),
            frozenset(),
            frozenset(),
            tuple(sorted(set(notes + ["no_extractable_change_event"]))),
        )
    return VisualSemanticFacts(
        1,
        frozenset(event[0] for event in events),
        frozenset(event[1] for event in events),
        frozenset(events),
        tuple(sorted(set(notes))),
    )


def _parse_gold_events(value: str) -> frozenset[tuple[str, str]]:
    """Parse explicit ``object:direction|...`` image-audited events."""

    text = value.strip()
    if not text:
        return frozenset()
    events: set[tuple[str, str]] = set()
    for item in text.split("|"):
        object_name, separator, direction = item.strip().partition(":")
        if not separator or object_name not in OBJECT_LABELS or direction not in DIRECTION_LABELS:
            raise ValueError(f"invalid gold_change_events entry: {item!r}")
        events.add((object_name, direction))
    return frozenset(events)


def parse_gold_semantics(row: dict[str, str]) -> VisualSemanticFacts | None:
    """Validate an image-audited gold row; U rows are retained but not scored."""

    label = row.get("gold_change_label", "").strip()
    if label == "U":
        return None
    if label not in {"0", "1"}:
        raise ValueError(f"invalid gold_change_label: {label!r}")
    objects = frozenset(item for item in row.get("gold_changed_objects", "").split("|") if item)
    directions = frozenset(
        item for item in row.get("gold_change_directions", "").split("|") if item
    )
    events = _parse_gold_events(row.get("gold_change_events", ""))
    if label == "0":
        if objects != {"none"} or directions != {"none"} or events:
            raise ValueError("gold label 0 requires objects=none, directions=none and no events")
        return VisualSemanticFacts(0, frozenset(), frozenset(), frozenset())
    if (
        not objects
        or not directions
        or not events
        or not objects <= OBJECT_LABELS
        or not directions <= DIRECTION_LABELS
    ):
        raise ValueError("gold label 1 contains invalid or missing object/direction/event labels")
    if objects != frozenset(object_name for object_name, _direction in events):
        raise ValueError("gold_changed_objects must exactly match objects in gold_change_events")
    if directions != frozenset(direction for _object_name, direction in events):
        raise ValueError(
            "gold_change_directions must exactly match directions in gold_change_events"
        )
    return VisualSemanticFacts(1, objects, directions, events)


def _set_counts(predicted: set[Any], gold: set[Any]) -> tuple[int, int, int]:
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sample_semantic_metrics(
    prediction: VisualSemanticFacts, gold: VisualSemanticFacts
) -> dict[str, int | bool]:
    """Calculate object, direction and object-direction event matches for one pair."""

    object_tp, object_fp, object_fn = _set_counts(set(prediction.objects), set(gold.objects))
    direction_tp, direction_fp, direction_fn = _set_counts(
        set(prediction.directions), set(gold.directions)
    )
    event_tp, event_fp, event_fn = _set_counts(set(prediction.events), set(gold.events))
    opposite = sum(
        (object_name, OPPOSITE_DIRECTIONS.get(direction, "")) in prediction.events
        for object_name, direction in gold.events
        if direction in OPPOSITE_DIRECTIONS
    )
    return {
        "binary_parse_success": prediction.change_label in {0, 1},
        "binary_correct": (
            prediction.change_label == gold.change_label
            if prediction.change_label in {0, 1}
            else False
        ),
        "object_tp": object_tp,
        "object_fp": object_fp,
        "object_fn": object_fn,
        "object_exact_match": prediction.objects == gold.objects,
        "direction_tp": direction_tp,
        "direction_fp": direction_fp,
        "direction_fn": direction_fn,
        "direction_exact_match": prediction.directions == gold.directions,
        "event_tp": event_tp,
        "event_fp": event_fp,
        "event_fn": event_fn,
        "event_exact_match": prediction.events == gold.events,
        "opposite_temporal_error_count": opposite,
    }


def aggregate_visual_semantic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate image-grounded gold comparisons with transparent denominators."""

    total = len(rows)
    resolved_rows = [row for row in rows if row["prediction"]["change_label"] in {0, 1}]
    unresolved_rows = [row for row in rows if row["prediction"]["change_label"] not in {0, 1}]
    tp = sum(
        row["gold"]["change_label"] == 1 and row["prediction"]["change_label"] == 1
        for row in resolved_rows
    )
    tn = sum(
        row["gold"]["change_label"] == 0 and row["prediction"]["change_label"] == 0
        for row in resolved_rows
    )
    fp = sum(
        row["gold"]["change_label"] == 0 and row["prediction"]["change_label"] == 1
        for row in resolved_rows
    )
    fn = sum(
        row["gold"]["change_label"] == 1 and row["prediction"]["change_label"] == 0
        for row in resolved_rows
    )
    unresolved = len(unresolved_rows)
    positive_rows = [row for row in rows if row["gold"]["change_label"] == 1]
    object_scores = _prf(
        sum(row["sample_metrics"]["object_tp"] for row in positive_rows),
        sum(row["sample_metrics"]["object_fp"] for row in positive_rows),
        sum(row["sample_metrics"]["object_fn"] for row in positive_rows),
    )
    direction_scores = _prf(
        sum(row["sample_metrics"]["direction_tp"] for row in positive_rows),
        sum(row["sample_metrics"]["direction_fp"] for row in positive_rows),
        sum(row["sample_metrics"]["direction_fn"] for row in positive_rows),
    )
    event_scores = _prf(
        sum(row["sample_metrics"]["event_tp"] for row in positive_rows),
        sum(row["sample_metrics"]["event_fp"] for row in positive_rows),
        sum(row["sample_metrics"]["event_fn"] for row in positive_rows),
    )
    recall_change = tp / (tp + fn) if tp + fn else None
    recall_no_change = tn / (tn + fp) if tn + fp else None
    return {
        "num_scored_samples": total,
        "binary": {
            "decision_coverage": len(resolved_rows) / total if total else 0.0,
            "resolved_sample_count": len(resolved_rows),
            "unresolved_prediction_count": unresolved,
            "unresolved_gold_change_count": sum(
                row["gold"]["change_label"] == 1 for row in unresolved_rows
            ),
            "unresolved_gold_no_change_count": sum(
                row["gold"]["change_label"] == 0 for row in unresolved_rows
            ),
            # Standard binary metrics use only samples with a persisted 0/1
            # decision.  The conservative all-sample score keeps abstentions
            # visible without pretending they are false-negative predictions.
            "accuracy": (tp + tn) / len(resolved_rows) if resolved_rows else None,
            "all_sample_accuracy_with_unresolved_as_incorrect": (
                (tp + tn) / total if total else 0.0
            ),
            "parse_success_rate": len(resolved_rows) / total if total else 0.0,
            "balanced_accuracy": (
                (recall_change + recall_no_change) / 2
                if recall_change is not None and recall_no_change is not None
                else None
            ),
            **_prf(tp, fp, fn),
            "no_change_recall": recall_no_change,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        },
        "object": {
            **object_scores,
            "num_positive_gold_samples": len(positive_rows),
            "exact_match_accuracy": (
                sum(bool(row["sample_metrics"]["object_exact_match"]) for row in positive_rows)
                / len(positive_rows)
                if positive_rows
                else 0.0
            ),
        },
        "direction": {
            **direction_scores,
            "num_positive_gold_samples": len(positive_rows),
            "exact_match_accuracy": (
                sum(bool(row["sample_metrics"]["direction_exact_match"]) for row in positive_rows)
                / len(positive_rows)
                if positive_rows
                else 0.0
            ),
            "opposite_temporal_error_count": sum(
                row["sample_metrics"]["opposite_temporal_error_count"] for row in positive_rows
            ),
        },
        "object_direction_event": {
            **event_scores,
            "num_positive_gold_samples": len(positive_rows),
            "exact_match_accuracy": (
                sum(bool(row["sample_metrics"]["event_exact_match"]) for row in positive_rows)
                / len(positive_rows)
                if positive_rows
                else 0.0
            ),
        },
        "gold_object_distribution": dict(
            sorted(
                Counter(
                    object_name for row in positive_rows for object_name in row["gold"]["objects"]
                ).items()
            )
        ),
        "gold_direction_distribution": dict(
            sorted(
                Counter(
                    direction for row in positive_rows for direction in row["gold"]["directions"]
                ).items()
            )
        ),
    }
