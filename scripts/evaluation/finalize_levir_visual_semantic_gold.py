"""Freeze dual-annotated LEVIR-CC visual semantic gold with required adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.validate_levir_visual_semantic_annotations import (  # noqa: E402
    _load_contract,
    validate_rows,
)

GOLD_FIELDS = [
    "sample_id",
    "image_t1_path",
    "image_t2_path",
    "gold_change_label",
    "gold_changed_objects",
    "gold_change_directions",
    "gold_change_events",
    "annotation_confidence",
    "label_source",
    "annotation_note",
]
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument(
        "--adjudicated",
        type=Path,
        required=True,
        help="Completed disagreement sheet from prepare_levir_visual_semantic_adjudication.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = validate_rows(list(csv.DictReader(file)), _load_contract())
    return {str(row["audit_id"]): row for row in rows}


def _read_adjudications(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    required = {
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
    if not required.issubset(set(reader.fieldnames or ())):
        raise ValueError(f"adjudication CSV missing columns: {sorted(required)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id", "")).strip()
        if not audit_id or audit_id in result:
            raise ValueError(f"missing or duplicate adjudication audit_id: {audit_id!r}")
        result[audit_id] = {key: str(value or "").strip() for key, value in row.items()}
    return result


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


def _tags(values: list[str]) -> str:
    return "|".join(sorted(values))


def _lower_confidence(left: str, right: str) -> str:
    return left if _CONFIDENCE_ORDER[left] <= _CONFIDENCE_ORDER[right] else right


def _validate_adjudicated_row(row: dict[str, str], audit_id: str) -> dict[str, Any]:
    normalized = validate_rows(
        [
            {
                "audit_id": audit_id,
                "sample_id": row["sample_id"],
                "image_t1_path": row["image_t1_path"],
                "image_t2_path": row["image_t2_path"],
                "gold_change_label": row["gold_change_label"],
                "gold_changed_objects": row["gold_changed_objects"],
                "gold_change_directions": row["gold_change_directions"],
                "gold_change_events": row["gold_change_events"],
                "annotation_confidence": row["annotation_confidence"],
                "annotation_note": row.get("adjudication_note", ""),
            }
        ],
        _load_contract(),
    )
    return normalized[0]


def build_gold_rows(
    annotator_a: dict[str, dict[str, Any]],
    annotator_b: dict[str, dict[str, Any]],
    adjudications: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    if set(annotator_a) != set(annotator_b):
        raise ValueError("annotator A and B audit ID sets must match exactly")
    if not set(adjudications).issubset(annotator_a):
        raise ValueError("adjudication IDs must be a subset of annotator IDs")
    result: list[dict[str, str]] = []
    agreed = adjudicated = 0
    for audit_id in sorted(annotator_a):
        left, right = annotator_a[audit_id], annotator_b[audit_id]
        for field in ("sample_id", "image_t1_path", "image_t2_path"):
            if left[field] != right[field]:
                raise ValueError(f"{field} mismatch for {audit_id}")
        if _same_semantics(left, right):
            chosen = left
            confidence = _lower_confidence(
                str(left["annotation_confidence"]), str(right["annotation_confidence"])
            )
            source, note = "dual_annotator_agreement", ""
            agreed += 1
        else:
            adjudication = adjudications.get(audit_id)
            if adjudication is None:
                raise ValueError(f"missing adjudication for disagreement: {audit_id}")
            for field in ("sample_id", "image_t1_path", "image_t2_path"):
                if adjudication.get(field) != left[field]:
                    raise ValueError(f"adjudication {field} mismatch for {audit_id}")
            chosen = _validate_adjudicated_row(adjudication, audit_id)
            confidence = str(chosen["annotation_confidence"])
            source, note = "third_annotator_adjudication", str(chosen["annotation_note"])
            adjudicated += 1
        result.append(
            {
                "sample_id": str(chosen["sample_id"]),
                "image_t1_path": str(chosen["image_t1_path"]),
                "image_t2_path": str(chosen["image_t2_path"]),
                "gold_change_label": str(chosen["gold_change_label"]),
                "gold_changed_objects": _tags(chosen["gold_changed_objects"]),
                "gold_change_directions": _tags(chosen["gold_change_directions"]),
                "gold_change_events": _tags(chosen["gold_change_events"]),
                "annotation_confidence": confidence,
                "label_source": source,
                "annotation_note": note,
            }
        )
    return result, {"num_agreement_rows": agreed, "num_adjudicated_rows": adjudicated}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    source_a, source_b, source_adjudication = (
        args.annotator_a.resolve(),
        args.annotator_b.resolve(),
        args.adjudicated.resolve(),
    )
    try:
        rows, counts = build_gold_rows(
            _read_annotations(source_a),
            _read_annotations(source_b),
            _read_adjudications(source_adjudication),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"visual semantic gold finalization failed: {exc}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "visual_semantic_gold_standard.csv"
    _write_csv(gold_path, rows)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "levir_cc_visual_change_semantics_v1_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_gold_rows": len(rows),
        **counts,
        "inputs": {
            "annotator_a": {"path": str(source_a), "sha256": _sha256(source_a)},
            "annotator_b": {"path": str(source_b), "sha256": _sha256(source_b)},
            "adjudicated": {
                "path": str(source_adjudication),
                "sha256": _sha256(source_adjudication),
            },
        },
        "output": {"path": str(gold_path), "sha256": _sha256(gold_path)},
    }
    (output_dir / "visual_semantic_gold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} frozen visual-semantic gold rows to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
