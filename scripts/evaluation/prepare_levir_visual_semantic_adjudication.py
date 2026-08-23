"""Create a focused adjudication sheet for disagreements in visual semantic labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.validate_levir_visual_semantic_annotations import (  # noqa: E402
    _load_contract,
    validate_rows,
)

OUTPUT_FIELDS = [
    "audit_id",
    "sample_id",
    "image_t1_path",
    "image_t2_path",
    "annotator_a_label",
    "annotator_a_objects",
    "annotator_a_directions",
    "annotator_a_events",
    "annotator_b_label",
    "annotator_b_objects",
    "annotator_b_directions",
    "annotator_b_events",
    "gold_change_label",
    "gold_changed_objects",
    "gold_change_directions",
    "gold_change_events",
    "annotation_confidence",
    "adjudication_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = validate_rows(list(csv.DictReader(file)), _load_contract())
    return {str(row["audit_id"]): row for row in rows}


def _pipe(values: list[str]) -> str:
    return "|".join(values)


def _same_semantics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[field] == right[field]
        for field in (
            "gold_change_label",
            "gold_changed_objects",
            "gold_change_directions",
            "gold_change_events",
        )
    )


def build_adjudication_rows(
    annotator_a: dict[str, dict[str, Any]], annotator_b: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    if set(annotator_a) != set(annotator_b):
        raise ValueError("annotator A and B audit ID sets must match exactly")
    output: list[dict[str, str]] = []
    for audit_id in sorted(annotator_a):
        left, right = annotator_a[audit_id], annotator_b[audit_id]
        for field in ("sample_id", "image_t1_path", "image_t2_path"):
            if left[field] != right[field]:
                raise ValueError(f"{field} mismatch for {audit_id}")
        if _same_semantics(left, right):
            continue
        output.append(
            {
                "audit_id": audit_id,
                "sample_id": str(left["sample_id"]),
                "image_t1_path": str(left["image_t1_path"]),
                "image_t2_path": str(left["image_t2_path"]),
                "annotator_a_label": str(left["gold_change_label"]),
                "annotator_a_objects": _pipe(left["gold_changed_objects"]),
                "annotator_a_directions": _pipe(left["gold_change_directions"]),
                "annotator_a_events": _pipe(left["gold_change_events"]),
                "annotator_b_label": str(right["gold_change_label"]),
                "annotator_b_objects": _pipe(right["gold_changed_objects"]),
                "annotator_b_directions": _pipe(right["gold_change_directions"]),
                "annotator_b_events": _pipe(right["gold_change_events"]),
                "gold_change_label": "",
                "gold_changed_objects": "",
                "gold_change_directions": "",
                "gold_change_events": "",
                "annotation_confidence": "",
                "adjudication_note": "",
            }
        )
    return output


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    try:
        source_a, source_b = args.annotator_a.resolve(), args.annotator_b.resolve()
        rows = build_adjudication_rows(_read_rows(source_a), _read_rows(source_b))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"visual semantic adjudication preparation failed: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "1.0",
        "protocol": "levir_cc_visual_change_semantics_v1_1",
        "annotator_a": {"path": str(source_a), "sha256": _sha256(source_a)},
        "annotator_b": {"path": str(source_b), "sha256": _sha256(source_b)},
        "num_disagreements": len(rows),
        "output": str(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} visual semantic disagreements to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
