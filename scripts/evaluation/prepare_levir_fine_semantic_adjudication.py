"""Create a focused adjudication sheet from two LEVIR semantic annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "audit_id",
    "caption",
    "human_change_label",
    "issue_type",
    "annotator_a_objects",
    "annotator_a_directions",
    "annotator_a_confidence",
    "annotator_a_note",
    "annotator_b_objects",
    "annotator_b_directions",
    "annotator_b_confidence",
    "annotator_b_note",
    "adjudicated_objects",
    "adjudicated_directions",
    "adjudicated_confidence",
    "adjudication_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id", "")).strip()
        if not audit_id or audit_id in result:
            raise ValueError(f"missing or duplicate audit_id in {path}: {audit_id!r}")
        result[audit_id] = {key: str(value or "").strip() for key, value in row.items()}
    return result


def _normalise_tags(raw: str) -> str:
    return "|".join(sorted({item.strip() for item in raw.split("|") if item.strip()}))


def _format_issue(objects: str, directions: str) -> bool:
    object_tags = set(_normalise_tags(objects).split("|")) - {""}
    direction_tags = set(_normalise_tags(directions).split("|")) - {""}
    return (
        len(object_tags) > 1 and bool(object_tags & {"none", "unknown"})
    ) or (len(direction_tags) > 1 and bool(direction_tags & {"none", "unknown"}))


def build_adjudication_rows(
    annotator_a: dict[str, dict[str, str]],
    annotator_b: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return only changed captions with format issues or semantic disagreement."""

    if set(annotator_a) != set(annotator_b):
        missing_a = sorted(set(annotator_b) - set(annotator_a))
        missing_b = sorted(set(annotator_a) - set(annotator_b))
        raise ValueError(
            f"annotation ID sets differ: missing_a={missing_a[:20]}, missing_b={missing_b[:20]}"
        )
    rows: list[dict[str, str]] = []
    format_issues = object_agreements = direction_agreements = changed_rows = 0
    for audit_id in sorted(annotator_a):
        left, right = annotator_a[audit_id], annotator_b[audit_id]
        if left.get("caption") != right.get("caption"):
            raise ValueError(f"caption mismatch for {audit_id}")
        label_a = left.get("human_change_label", "")
        label_b = right.get("human_change_label", "")
        if label_a != label_b:
            raise ValueError(
                f"locked binary label mismatch for {audit_id}: {label_a!r} != {label_b!r}"
            )
        if label_a != "1":
            continue
        changed_rows += 1
        object_a = _normalise_tags(left.get("human_changed_objects", ""))
        object_b = _normalise_tags(right.get("human_changed_objects", ""))
        direction_a = _normalise_tags(left.get("human_change_directions", ""))
        direction_b = _normalise_tags(right.get("human_change_directions", ""))
        object_equal = object_a == object_b
        direction_equal = direction_a == direction_b
        object_agreements += int(object_equal)
        direction_agreements += int(direction_equal)
        malformed = _format_issue(object_a, direction_a) or _format_issue(object_b, direction_b)
        if not malformed and object_equal and direction_equal:
            continue
        issue_type = []
        if malformed:
            issue_type.append("format")
            format_issues += 1
        if not object_equal:
            issue_type.append("object_disagreement")
        if not direction_equal:
            issue_type.append("direction_disagreement")
        rows.append(
            {
                "audit_id": audit_id,
                "caption": left["caption"],
                "human_change_label": label_a,
                "issue_type": "|".join(issue_type),
                "annotator_a_objects": object_a,
                "annotator_a_directions": direction_a,
                "annotator_a_confidence": left.get("human_annotation_confidence", ""),
                "annotator_a_note": left.get("human_semantic_note", ""),
                "annotator_b_objects": object_b,
                "annotator_b_directions": direction_b,
                "annotator_b_confidence": right.get("human_annotation_confidence", ""),
                "annotator_b_note": right.get("human_semantic_note", ""),
                "adjudicated_objects": "",
                "adjudicated_directions": "",
                "adjudicated_confidence": "",
                "adjudication_note": "",
            }
        )
    summary = {
        "schema_version": "1.0",
        "protocol": "levir_cc_caption_semantics_v1",
        "num_changed_rows": changed_rows,
        "num_object_exact_agreements": object_agreements,
        "object_exact_agreement_rate": object_agreements / changed_rows if changed_rows else None,
        "num_direction_exact_agreements": direction_agreements,
        "direction_exact_agreement_rate": (
            direction_agreements / changed_rows if changed_rows else None
        ),
        "num_adjudication_rows": len(rows),
        "num_format_issue_rows": format_issues,
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    if output_csv.exists() or summary_json.exists():
        raise SystemExit("refusing to overwrite an existing adjudication output")
    try:
        rows, summary = build_adjudication_rows(
            _read_rows(args.annotator_a.resolve()), _read_rows(args.annotator_b.resolve())
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} semantic adjudication rows to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
