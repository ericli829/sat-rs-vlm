"""Prepare two independent LEVIR caption-semantic augmentation sheets.

The input is an already adjudicated binary caption-semantics gold CSV.  Its
binary label is deliberately copied as a locked field, so annotators only add
object and direction semantics.  This prevents relabelling the completed
binary gold standard while keeping the fine-grained labels independent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "levir_cc_fine_semantic_schema_v1.json"
)

OUTPUT_FIELDS = [
    "audit_id",
    "caption",
    "human_change_label",
    "human_changed_objects",
    "human_change_directions",
    "human_annotation_confidence",
    "human_semantic_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-csv",
        type=Path,
        required=True,
        help="Existing adjudicated CSV containing audit_id, caption and human_gold_label.",
    )
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_gold_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"audit_id", "caption", "human_gold_label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"gold CSV must contain columns: {sorted(required)}")
    seen: set[str] = set()
    for row in rows:
        audit_id = str(row.get("audit_id", "")).strip()
        label = str(row.get("human_gold_label", "")).strip()
        if not audit_id or audit_id in seen:
            raise ValueError(f"missing or duplicate audit_id: {audit_id!r}")
        if not str(row.get("caption", "")).strip():
            raise ValueError(f"empty caption for {audit_id}")
        if label not in {"0", "1", "U"}:
            raise ValueError(f"invalid human_gold_label for {audit_id}: {label!r}")
        seen.add(audit_id)
    return rows


def build_annotation_rows(gold_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build semantic sheets; only changed captions require new annotation."""

    result: list[dict[str, str]] = []
    for row in gold_rows:
        label = str(row["human_gold_label"]).strip()
        if label == "0":
            objects, directions, confidence = "none", "none", "high"
        elif label == "U":
            objects, directions, confidence = "unknown", "unknown", "low"
        else:
            objects, directions, confidence = "", "", ""
        result.append(
            {
                "audit_id": str(row["audit_id"]).strip(),
                "caption": str(row["caption"]).strip(),
                "human_change_label": label,
                "human_changed_objects": objects,
                "human_change_directions": directions,
                "human_annotation_confidence": confidence,
                "human_semantic_note": "",
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    *,
    source: Path,
    split: str,
    rows: list[dict[str, str]],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "levir_cc_caption_semantics_v1",
        "split": split,
        "source_gold_csv": str(source),
        "source_gold_csv_sha256": _sha256(source),
        "num_rows": len(rows),
        "num_changed_rows_requiring_annotation": sum(
            row["human_change_label"] == "1" for row in rows
        ),
        "num_locked_no_change_rows": sum(row["human_change_label"] == "0" for row in rows),
        "num_locked_uncertain_rows": sum(row["human_change_label"] == "U" for row in rows),
        "files": ["annotator_a_semantic.csv", "annotator_b_semantic.csv", "schema.json"],
        "warning": (
            "Do not use holdout annotations to change the local judge, prompt, rules or "
            "training data before the final frozen evaluation."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = args.gold_csv.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    rows = build_annotation_rows(_read_gold_rows(source))
    output_dir.mkdir(parents=True, exist_ok=True)
    for annotator in ("annotator_a", "annotator_b"):
        _write_csv(output_dir / f"{annotator}_semantic.csv", rows)
    shutil.copyfile(SCHEMA_PATH, output_dir / "schema.json")
    _write_manifest(
        output_dir / "semantic_audit_manifest.json",
        source=source,
        split=args.split,
        rows=rows,
    )
    print(f"Saved {len(rows)} semantic annotation rows to: {output_dir}")
    num_changed = sum(row["human_change_label"] == "1" for row in rows)
    print(f"Changed rows requiring object/direction labels: {num_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
