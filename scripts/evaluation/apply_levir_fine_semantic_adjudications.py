"""Apply reviewed LEVIR semantic adjudications without changing source sheets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.validate_levir_fine_semantic_annotations import (  # noqa: E402
    _load_schema,
    validate_rows,
)

DECISION_FIELDS = (
    "adjudicated_objects",
    "adjudicated_directions",
    "adjudicated_confidence",
    "adjudication_note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication-csv", type=Path, required=True)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _load_decisions(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), dict):
        raise ValueError("decisions JSON must contain a decisions object")
    decisions: dict[str, dict[str, str]] = {}
    for audit_id, raw in payload["decisions"].items():
        if not isinstance(audit_id, str) or not isinstance(raw, dict):
            raise ValueError("each decision must map an audit_id string to an object")
        if set(raw) != set(DECISION_FIELDS):
            raise ValueError(f"{audit_id}: decision fields must be exactly {list(DECISION_FIELDS)}")
        decisions[audit_id] = {field: str(raw[field]).strip() for field in DECISION_FIELDS}
    return decisions


def apply_decisions(
    rows: list[dict[str, str]], decisions: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    source_ids = {str(row.get("audit_id", "")).strip() for row in rows}
    if not source_ids or "" in source_ids:
        raise ValueError("adjudication CSV contains a missing audit_id")
    if source_ids != set(decisions):
        missing = sorted(source_ids - set(decisions))
        extra = sorted(set(decisions) - source_ids)
        raise ValueError(
            f"decisions must cover every row exactly; missing={missing[:20]}, extra={extra[:20]}"
        )
    completed: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    for row in rows:
        audit_id = str(row["audit_id"]).strip()
        result = dict(row)
        result.update(decisions[audit_id])
        completed.append(result)
        validation_rows.append(
            {
                "audit_id": audit_id,
                "caption": str(row.get("caption", "")),
                "human_change_label": str(row.get("human_change_label", "")),
                "human_changed_objects": result["adjudicated_objects"],
                "human_change_directions": result["adjudicated_directions"],
                "human_annotation_confidence": result["adjudicated_confidence"],
                "human_semantic_note": result["adjudication_note"],
            }
        )
    validate_rows(validation_rows, _load_schema())
    return completed


def main() -> int:
    args = parse_args()
    source = args.adjudication_csv.resolve()
    output = args.output_csv.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames
    if not fields:
        raise SystemExit("adjudication CSV has no header")
    try:
        completed = apply_decisions(rows, _load_decisions(args.decisions_json.resolve()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    print(f"Saved validated adjudications for {len(completed)} rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
