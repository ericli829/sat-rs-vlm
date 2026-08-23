"""Create a hash-traceable gold CSV aligned to another model's prediction IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--pairing-csv", type=Path, required=True)
    parser.add_argument("--source-id-column", required=True)
    parser.add_argument("--target-id-column", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    with args.pairing_csv.resolve().open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {args.source_id_column, args.target_id_column}
        if not required.issubset(set(reader.fieldnames or ())):
            raise SystemExit(f"pairing CSV is missing columns: {sorted(required)}")
        mapping: dict[str, str] = {}
        for row in reader:
            source_id = str(row.get(args.source_id_column, "")).strip()
            target_id = str(row.get(args.target_id_column, "")).strip()
            if not source_id or not target_id or source_id in mapping:
                raise SystemExit("pairing CSV contains missing or duplicate source IDs")
            mapping[source_id] = target_id
    with args.gold_csv.resolve().open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if "sample_id" not in fieldnames:
        raise SystemExit("gold CSV has no sample_id column")
    seen: set[str] = set()
    for row in rows:
        source_id = str(row.get("sample_id", "")).strip()
        target_id = mapping.get(source_id)
        if target_id is None:
            raise SystemExit(f"gold sample is absent from pairing CSV: {source_id}")
        if target_id in seen:
            raise SystemExit(f"target prediction ID is duplicated: {target_id}")
        row["sample_id"] = target_id
        seen.add(target_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aligned {len(rows)} gold rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
