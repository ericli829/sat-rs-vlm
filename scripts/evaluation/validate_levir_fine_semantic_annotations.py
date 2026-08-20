"""Validate completed LEVIR fine-semantic annotation CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "levir_cc_fine_semantic_schema_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _values(raw: str, allowed: set[str], field: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split("|") if value.strip())
    if not values:
        raise ValueError(f"{field} is empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicates")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"{field} contains invalid values: {invalid}")
    if ("none" in values or "unknown" in values) and len(values) != 1:
        raise ValueError(f"{field} cannot combine none/unknown with another value")
    return values


def validate_rows(rows: list[dict[str, str]], schema: dict[str, object]) -> list[dict[str, object]]:
    fields = schema["fields"]
    assert isinstance(fields, dict)
    objects_spec = fields["human_changed_objects"]
    directions_spec = fields["human_change_directions"]
    confidence_spec = fields["human_annotation_confidence"]
    assert isinstance(objects_spec, dict)
    assert isinstance(directions_spec, dict)
    assert isinstance(confidence_spec, dict)
    object_values = set(objects_spec["values"])
    direction_values = set(directions_spec["values"])
    confidence_values = set(confidence_spec["values"])
    seen: set[str] = set()
    normalized: list[dict[str, object]] = []
    for line, row in enumerate(rows, start=2):
        audit_id = str(row.get("audit_id", "")).strip()
        label = str(row.get("human_change_label", "")).strip()
        if not audit_id or audit_id in seen:
            raise ValueError(f"line {line}: missing or duplicate audit_id {audit_id!r}")
        if label not in {"0", "1", "U"}:
            raise ValueError(f"line {line}: invalid human_change_label {label!r}")
        objects = _values(str(row.get("human_changed_objects", "")), object_values, "objects")
        directions = _values(
            str(row.get("human_change_directions", "")), direction_values, "directions"
        )
        confidence = str(row.get("human_annotation_confidence", "")).strip()
        if confidence not in confidence_values:
            raise ValueError(f"line {line}: invalid confidence {confidence!r}")
        if label == "0" and (objects != ("none",) or directions != ("none",)):
            raise ValueError(f"line {line}: label 0 requires none/none")
        if label == "U" and (objects != ("unknown",) or directions != ("unknown",)):
            raise ValueError(f"line {line}: label U requires unknown/unknown")
        if label == "1" and (objects == ("none",) or directions == ("none",)):
            raise ValueError(
                f"line {line}: label 1 requires object and non-none direction semantics"
            )
        seen.add(audit_id)
        normalized.append(
            {
                "audit_id": audit_id,
                "caption": str(row.get("caption", "")).strip(),
                "human_change_label": label,
                "human_changed_objects": list(objects),
                "human_change_directions": list(directions),
                "human_annotation_confidence": confidence,
                "human_semantic_note": str(row.get("human_semantic_note", "")).strip(),
            }
        )
    return normalized


def main() -> int:
    args = parse_args()
    source = args.annotations.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    with source.open(encoding="utf-8-sig", newline="") as file:
        normalized = validate_rows(list(csv.DictReader(file)), _load_schema())
    summary = {
        "schema_version": "1.0",
        "protocol": "levir_cc_caption_semantics_v1",
        "source": str(source),
        "num_rows": len(normalized),
        "change_label_distribution": dict(Counter(row["human_change_label"] for row in normalized)),
        "object_label_distribution": dict(
            Counter(label for row in normalized for label in row["human_changed_objects"])
        ),
        "direction_label_distribution": dict(
            Counter(label for row in normalized for label in row["human_change_directions"])
        ),
        "status": "valid",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(normalized)} rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
