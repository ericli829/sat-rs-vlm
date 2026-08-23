"""Validate a completed LEVIR-CC image-level semantic annotation sheet."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "levir_cc_visual_semantic_contract_v1.json"
)

REQUIRED_COLUMNS = {
    "audit_id",
    "sample_id",
    "image_t1_path",
    "image_t2_path",
    "gold_change_label",
    "gold_changed_objects",
    "gold_change_directions",
    "gold_change_events",
    "annotation_confidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _values(raw: str, allowed: set[str], field: str, *, allow_empty: bool = False) -> set[str]:
    values = {value.strip() for value in raw.split("|") if value.strip()}
    if not values and not allow_empty:
        raise ValueError(f"{field} is empty")
    invalid = sorted(values - allowed)
    if invalid:
        raise ValueError(f"{field} contains invalid values: {invalid}")
    if ("none" in values or "unknown" in values) and len(values) != 1:
        raise ValueError(f"{field} cannot combine none/unknown with another value")
    return values


def _events(raw: str, objects: set[str], directions: set[str]) -> set[tuple[str, str]]:
    parsed: set[tuple[str, str]] = set()
    for item in (part.strip() for part in raw.split("|") if part.strip()):
        if item.count(":") != 1:
            raise ValueError(f"invalid event {item!r}; expected object:direction")
        object_name, direction = (part.strip() for part in item.split(":", maxsplit=1))
        if not object_name or not direction:
            raise ValueError(f"invalid empty event component in {item!r}")
        parsed.add((object_name, direction))
    if not parsed:
        raise ValueError("gold_change_events is empty for a changed row")
    event_objects, event_directions = {left for left, _ in parsed}, {right for _, right in parsed}
    if event_objects != objects or event_directions != directions:
        raise ValueError("event object/direction sets must exactly match the declared field sets")
    return parsed


def validate_rows(rows: list[dict[str, str]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows or not REQUIRED_COLUMNS.issubset(rows[0]):
        raise ValueError(f"annotation CSV is missing required columns: {sorted(REQUIRED_COLUMNS)}")
    labels = contract["labels"]
    object_values = set(labels["objects"])
    direction_values = set(labels["directions"])
    normalized: list[dict[str, Any]] = []
    seen_audit, seen_sample = set(), set()
    for line, row in enumerate(rows, start=2):
        audit_id = str(row.get("audit_id", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        if not audit_id or audit_id in seen_audit or not sample_id or sample_id in seen_sample:
            raise ValueError(f"line {line}: audit_id and sample_id must be non-empty and unique")
        t1, t2 = (
            str(row.get("image_t1_path", "")).strip(),
            str(row.get("image_t2_path", "")).strip(),
        )
        if not t1 or not t2:
            raise ValueError(f"line {line}: before/after image paths must be non-empty")
        label = str(row.get("gold_change_label", "")).strip()
        if label not in {"0", "1", "U"}:
            raise ValueError(f"line {line}: gold_change_label must be 0, 1 or U")
        objects = _values(str(row.get("gold_changed_objects", "")), object_values, "objects")
        directions = _values(
            str(row.get("gold_change_directions", "")), direction_values, "directions"
        )
        confidence = str(row.get("annotation_confidence", "")).strip()
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"line {line}: annotation_confidence must be high, medium or low")
        raw_events = str(row.get("gold_change_events", "")).strip()
        if label == "0":
            if objects != {"none"} or directions != {"none"} or raw_events:
                raise ValueError(f"line {line}: no-change requires none/none and empty events")
            events: set[tuple[str, str]] = set()
        elif label == "U":
            if objects != {"unknown"} or directions != {"unknown"} or raw_events:
                raise ValueError(
                    f"line {line}: uncertain requires unknown/unknown and empty events"
                )
            events = set()
        else:
            if {"none", "unknown"} & objects or {"none", "unknown"} & directions:
                raise ValueError(f"line {line}: changed row cannot use none/unknown labels")
            events = _events(raw_events, objects, directions)
        seen_audit.add(audit_id)
        seen_sample.add(sample_id)
        normalized.append(
            {
                "audit_id": audit_id,
                "sample_id": sample_id,
                "image_t1_path": t1,
                "image_t2_path": t2,
                "gold_change_label": label,
                "gold_changed_objects": sorted(objects),
                "gold_change_directions": sorted(directions),
                "gold_change_events": [f"{left}:{right}" for left, right in sorted(events)],
                "annotation_confidence": confidence,
                "annotation_note": str(row.get("annotation_note", "")).strip(),
            }
        )
    return normalized


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    source, output = args.annotations.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    try:
        with source.open(encoding="utf-8-sig", newline="") as file:
            normalized = validate_rows(list(csv.DictReader(file)), _load_contract())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"visual annotation validation failed: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "protocol": "levir_cc_visual_change_semantics_v1_1",
        "source": str(source),
        "num_rows": len(normalized),
        "change_label_distribution": dict(Counter(row["gold_change_label"] for row in normalized)),
        "status": "valid",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(normalized)} visual annotation rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
