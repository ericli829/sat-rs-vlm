"""Apply explicitly supplied adjudication corrections without altering the source CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--resolutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, resolutions_path, output = (
        args.adjudication.resolve(),
        args.resolutions.resolve(),
        args.output.resolve(),
    )
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    try:
        resolutions = json.loads(resolutions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read resolutions JSON: {exc}") from exc
    if not isinstance(resolutions, list) or not resolutions:
        raise SystemExit("resolutions JSON must be a non-empty list")
    by_id: dict[str, dict[str, str]] = {}
    for item in resolutions:
        if not isinstance(item, dict):
            raise SystemExit("each resolution must be an object")
        audit_id = str(item.get("audit_id", "")).strip()
        if not audit_id or audit_id in by_id:
            raise SystemExit(f"missing or duplicate resolution audit_id: {audit_id!r}")
        updates = {
            key: str(value).strip()
            for key, value in item.items()
            if key
            in {
                "gold_change_label",
                "gold_changed_objects",
                "gold_change_directions",
                "gold_change_events",
                "annotation_confidence",
                "adjudication_note",
            }
        }
        if not updates:
            raise SystemExit(f"resolution {audit_id} contains no editable adjudication fields")
        by_id[audit_id] = updates

    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if "audit_id" not in fieldnames:
        raise SystemExit("adjudication CSV has no audit_id column")
    seen: set[str] = set()
    for row in rows:
        audit_id = str(row.get("audit_id", "")).strip()
        if audit_id in by_id:
            row.update(by_id[audit_id])
            seen.add(audit_id)
    missing = sorted(set(by_id) - seen)
    if missing:
        raise SystemExit(f"resolution audit IDs are absent from source CSV: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Applied {len(seen)} adjudication resolutions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
