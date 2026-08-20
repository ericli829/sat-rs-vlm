"""Finalize LEVIR fine-semantic development gold from dual labels and adjudications."""

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

from scripts.evaluation.validate_levir_fine_semantic_annotations import (  # noqa: E402
    _load_schema,
    validate_rows,
)

GOLD_FIELDS = [
    "audit_id",
    "caption",
    "human_change_label",
    "human_changed_objects",
    "human_change_directions",
    "human_annotation_confidence",
    "label_source",
    "human_semantic_note",
]
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument(
        "--adjudication-provenance",
        default="human_third_annotator",
        help="Provenance for adjudicated rows, recorded verbatim in the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id", "")).strip()
        if not audit_id or audit_id in result:
            raise ValueError(f"missing or duplicate audit_id in {path}: {audit_id!r}")
        result[audit_id] = {key: str(value or "").strip() for key, value in row.items()}
    return result


def _tags(raw: str) -> str:
    return "|".join(sorted({item.strip() for item in raw.split("|") if item.strip()}))


def _lower_confidence(left: str, right: str) -> str:
    if left not in _CONFIDENCE_ORDER or right not in _CONFIDENCE_ORDER:
        raise ValueError(f"invalid confidence values: {left!r}, {right!r}")
    return left if _CONFIDENCE_ORDER[left] <= _CONFIDENCE_ORDER[right] else right


def build_gold_rows(
    annotator_a: dict[str, dict[str, str]],
    annotator_b: dict[str, dict[str, str]],
    adjudications: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Merge dual annotations and require an adjudication for every semantic disagreement."""

    if set(annotator_a) != set(annotator_b):
        raise ValueError("annotator A and B ID sets must match exactly")
    if not set(adjudications).issubset(annotator_a):
        raise ValueError("adjudication IDs must be a subset of annotator IDs")
    gold_rows: list[dict[str, str]] = []
    agreed = adjudicated = 0
    for audit_id in sorted(annotator_a):
        left, right = annotator_a[audit_id], annotator_b[audit_id]
        if left.get("caption") != right.get("caption"):
            raise ValueError(f"caption mismatch for {audit_id}")
        label = left.get("human_change_label", "")
        if label != right.get("human_change_label", ""):
            raise ValueError(f"binary label mismatch for {audit_id}")
        object_a, object_b = _tags(left.get("human_changed_objects", "")), _tags(
            right.get("human_changed_objects", "")
        )
        direction_a, direction_b = _tags(left.get("human_change_directions", "")), _tags(
            right.get("human_change_directions", "")
        )
        needs_adjudication = label == "1" and (
            object_a != object_b or direction_a != direction_b or audit_id in adjudications
        )
        if needs_adjudication:
            row = adjudications.get(audit_id)
            if row is None:
                raise ValueError(f"missing adjudication for semantic disagreement: {audit_id}")
            objects = _tags(row.get("adjudicated_objects", ""))
            directions = _tags(row.get("adjudicated_directions", ""))
            confidence = row.get("adjudicated_confidence", "")
            note = row.get("adjudication_note", "")
            source = "adjudicated"
            adjudicated += 1
        else:
            objects, directions = object_a, direction_a
            confidence = _lower_confidence(
                left.get("human_annotation_confidence", ""),
                right.get("human_annotation_confidence", ""),
            )
            note = ""
            source = "annotator_agreement"
            agreed += 1
        gold_rows.append(
            {
                "audit_id": audit_id,
                "caption": left["caption"],
                "human_change_label": label,
                "human_changed_objects": objects,
                "human_change_directions": directions,
                "human_annotation_confidence": confidence,
                "label_source": source,
                "human_semantic_note": note,
            }
        )
    validate_rows(gold_rows, _load_schema())
    return gold_rows, {"num_agreement_rows": agreed, "num_adjudicated_rows": adjudicated}


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
            _read_csv(source_a), _read_csv(source_b), _read_csv(source_adjudication)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "human_fine_semantic_gold_standard.csv", rows)
    with (output_dir / "human_fine_semantic_gold_standard.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "levir_cc_caption_semantics_v1",
        "split": "development",
        "num_gold_rows": len(rows),
        **counts,
        "adjudication_provenance": str(args.adjudication_provenance),
        "inputs": {
            "annotator_a": {"path": str(source_a), "sha256": _sha256(source_a)},
            "annotator_b": {"path": str(source_b), "sha256": _sha256(source_b)},
            "adjudicated": {
                "path": str(source_adjudication),
                "sha256": _sha256(source_adjudication),
            },
        },
        "files": [
            "human_fine_semantic_gold_standard.csv",
            "human_fine_semantic_gold_standard.jsonl",
        ],
        "limitation": "Caption-semantic gold only; not image-level factual ground truth.",
    }
    (output_dir / "fine_semantic_gold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} finalized fine-semantic gold rows to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
