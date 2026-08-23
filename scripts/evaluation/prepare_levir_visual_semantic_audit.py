"""Prepare blinded dual-annotator sheets for LEVIR-CC image-level semantics.

The source mapping may contain split/changeflag metadata for deterministic sampling,
but the generated sheets deliberately expose only the before/after image pair and
empty visual-semantic fields.  They never expose model captions, reference captions
or dataset changeflag values to annotators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTPUT_FIELDS = [
    "audit_id",
    "sample_id",
    "image_t1_path",
    "image_t2_path",
    "gold_change_label",
    "gold_changed_objects",
    "gold_change_directions",
    "gold_change_events",
    "annotation_confidence",
    "annotation_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-mapping",
        type=Path,
        required=True,
        help="CSV with id/sample_id, image_t1_path and image_t2_path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional deterministic subset size. Omit to use every mapping row.",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--split",
        help="Optional split value to retain, for example val or test.",
    )
    parser.add_argument(
        "--verify-image-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require mapped local image files to exist (default: true).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_path(mapping_path: Path, raw: str) -> Path:
    candidate = Path(raw.strip())
    return candidate if candidate.is_absolute() else mapping_path.parent / candidate


def read_mapping(
    path: Path, *, split: str | None, verify_image_paths: bool
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("image mapping is empty")
    fieldnames = set(rows[0])
    id_field = "sample_id" if "sample_id" in fieldnames else "id" if "id" in fieldnames else None
    required = {"image_t1_path", "image_t2_path"}
    if id_field is None or not required.issubset(fieldnames):
        raise ValueError("image mapping requires id/sample_id, image_t1_path and image_t2_path")
    if split is not None and "split" not in fieldnames:
        raise ValueError("--split requires a split column in image mapping")

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        if split is not None and str(row.get("split", "")).strip() != split:
            continue
        sample_id = str(row.get(id_field, "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"line {line}: missing or duplicate sample id {sample_id!r}")
        t1, t2 = (
            str(row.get("image_t1_path", "")).strip(),
            str(row.get("image_t2_path", "")).strip(),
        )
        if not t1 or not t2:
            raise ValueError(f"line {line}: both image paths are required")
        if verify_image_paths:
            for field, raw in (("image_t1_path", t1), ("image_t2_path", t2)):
                if not _image_path(path, raw).is_file():
                    raise ValueError(f"line {line}: {field} is not a readable local file")
        seen.add(sample_id)
        result.append({"sample_id": sample_id, "image_t1_path": t1, "image_t2_path": t2})
    if not result:
        raise ValueError("no mapping rows remained after split filtering")
    return result


def build_annotation_rows(
    mapping_rows: list[dict[str, str]], *, sample_size: int | None, seed: int
) -> list[dict[str, str]]:
    if sample_size is not None and not 1 <= sample_size <= len(mapping_rows):
        raise ValueError("--sample-size must be between 1 and the number of mapping rows")
    selected = list(mapping_rows)
    random.Random(seed).shuffle(selected)
    if sample_size is not None:
        selected = selected[:sample_size]
    return [
        {
            "audit_id": f"visual_{index:04d}",
            **row,
            "gold_change_label": "",
            "gold_changed_objects": "",
            "gold_change_directions": "",
            "gold_change_events": "",
            "annotation_confidence": "",
            "annotation_note": "",
        }
        for index, row in enumerate(selected, start=1)
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source, output_dir = args.image_mapping.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    try:
        mapping_rows = read_mapping(
            source, split=args.split, verify_image_paths=args.verify_image_paths
        )
        rows = build_annotation_rows(mapping_rows, sample_size=args.sample_size, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"visual semantic audit preparation failed: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for annotator in ("annotator_a", "annotator_b"):
        _write_csv(output_dir / f"{annotator}_visual_semantic.csv", rows)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "levir_cc_visual_change_semantics_v1_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_mapping": {"path": str(source), "sha256": _sha256(source)},
        "num_mapping_rows": len(mapping_rows),
        "num_annotation_rows": len(rows),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "split": args.split,
        "verify_image_paths": args.verify_image_paths,
        "blinding": {
            "not_exposed": ["prediction", "reference", "changeflag"],
            "shown": ["sample_id", "image_t1_path", "image_t2_path"],
        },
        "files": ["annotator_a_visual_semantic.csv", "annotator_b_visual_semantic.csv"],
    }
    (output_dir / "visual_semantic_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} blinded visual-semantic rows to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
