"""Freeze a paired LEVIR-CC image subset and extract only its image pairs.

This utility is intentionally offline and standard-library only.  It selects
one record per image pair from a primary prediction file (normally the sampled
4B run), finds the matching image-pair record in a second prediction file
(normally the 2B full run), and writes blinded-image inputs for human audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--paired-predictions", type=Path, required=True)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def levir_rows(path: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(path)
        if row.get("task_type") == "change_detection"
        and isinstance(row.get("metadata"), dict)
        and row["metadata"].get("dataset") == "LEVIR-CC"
    ]


def annotation_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in levir_rows(path):
        sample_id = str(row.get("id", "")).strip()
        images = row.get("images")
        metadata = row.get("metadata")
        if not sample_id or not isinstance(images, list) or len(images) != 2:
            raise ValueError(f"annotation {sample_id!r} does not contain exactly two images")
        if not isinstance(metadata, dict):
            raise ValueError(f"annotation {sample_id!r} has no metadata")
        if sample_id in index:
            raise ValueError(f"duplicate annotation id: {sample_id}")
        index[sample_id] = row
    return index


def pair_key(annotation: dict[str, Any]) -> str:
    images = annotation["images"]
    return "|".join(Path(str(image)).name for image in images)


def select_rows(
    primary: list[dict[str, Any]],
    annotation_by_id: dict[str, dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size < 2 or sample_size % 2:
        raise ValueError(
            "--sample-size must be an even integer of at least 2 for balanced changeflag sampling"
        )
    primary_by_pair: dict[str, dict[str, Any]] = {}
    for row in primary:
        sample_id = str(row.get("id", "")).strip()
        annotation = annotation_by_id.get(sample_id)
        if annotation is None:
            raise ValueError(f"primary prediction id missing from annotations: {sample_id}")
        key = pair_key(annotation)
        if key in primary_by_pair:
            raise ValueError(f"primary predictions contain duplicate image pair: {key}")
        primary_by_pair[key] = row

    by_flag: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in primary_by_pair.values():
        flag = annotation_by_id[str(row["id"])]["metadata"].get("changeflag")
        if flag not in by_flag:
            raise ValueError(f"primary id {row['id']} has invalid changeflag: {flag!r}")
        by_flag[flag].append(row)
    per_flag = sample_size // 2
    if any(len(rows) < per_flag for rows in by_flag.values()):
        raise ValueError("insufficient primary rows for balanced frozen subset")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for flag in (0, 1):
        rows = sorted(by_flag[flag], key=lambda row: str(row["id"]))
        rng.shuffle(rows)
        selected.extend(rows[:per_flag])
    return sorted(selected, key=lambda row: str(row["id"]))


def copy_member(archive: zipfile.ZipFile, member_name: str, destination: Path) -> None:
    try:
        source = archive.open(member_name)
    except KeyError as exc:
        raise ValueError(f"dataset ZIP does not contain image: {member_name}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    annotations = args.annotations.resolve()
    primary_path = args.primary_predictions.resolve()
    paired_path = args.paired_predictions.resolve()
    archive_path = args.dataset_zip.resolve()
    for path in (annotations, primary_path, paired_path, archive_path):
        if not path.is_file():
            raise SystemExit(f"input file not found: {path}")

    try:
        annotation_by_id = annotation_index(annotations)
        primary = levir_rows(primary_path)
        paired = levir_rows(paired_path)
        selected = select_rows(
            primary, annotation_by_id, sample_size=args.sample_size, seed=args.seed
        )
        paired_by_key: dict[str, dict[str, Any]] = {}
        for row in paired:
            sample_id = str(row.get("id", "")).strip()
            annotation = annotation_by_id.get(sample_id)
            if annotation is None:
                raise ValueError(f"paired prediction id missing from annotations: {sample_id}")
            key = pair_key(annotation)
            if key in paired_by_key:
                raise ValueError(f"paired predictions contain duplicate image pair: {key}")
            paired_by_key[key] = row
        missing = [
            row["id"]
            for row in selected
            if pair_key(annotation_by_id[row["id"]]) not in paired_by_key
        ]
        if missing:
            raise ValueError(f"paired predictions lack {len(missing)} selected image pairs")
    except ValueError as exc:
        raise SystemExit(f"frozen subset preparation failed: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, str]] = []
    pairing_rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for row in selected:
            primary_id = str(row["id"])
            annotation = annotation_by_id[primary_id]
            key = pair_key(annotation)
            paired_id = str(paired_by_key[key]["id"])
            before_name, after_name = (Path(str(raw)).name for raw in annotation["images"])
            split = str(annotation["metadata"]["split"])
            zip_before = f"images/{split}/A/{before_name}"
            zip_after = f"images/{split}/B/{after_name}"
            before_rel = Path("images") / "before" / before_name
            after_rel = Path("images") / "after" / after_name
            copy_member(archive, zip_before, output_dir / before_rel)
            copy_member(archive, zip_after, output_dir / after_rel)
            mapping_rows.append(
                {
                    "sample_id": primary_id,
                    # Annotation sheets are written below a child directory.
                    # Use absolute paths so their image references remain valid
                    # when an annotator opens a CSV from that child directory.
                    "image_t1_path": str((output_dir / before_rel).resolve()),
                    "image_t2_path": str((output_dir / after_rel).resolve()),
                    "split": split,
                    "image_pair_key": key,
                }
            )
            pairing_rows.append(
                {
                    "image_pair_key": key,
                    "primary_prediction_id": primary_id,
                    "paired_prediction_id": paired_id,
                    "split": split,
                    "changeflag_for_sampling_only": str(annotation["metadata"]["changeflag"]),
                }
            )

    mapping_path = output_dir / "image_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    pairing_path = output_dir / "model_pairing_private.csv"
    with pairing_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairing_rows[0]))
        writer.writeheader()
        writer.writerows(pairing_rows)
    flag_counts = Counter(row["changeflag_for_sampling_only"] for row in pairing_rows)
    manifest = {
        "schema_version": "1.0",
        "status": "historical_exploratory_gold_preparation",
        "purpose": "blind image-level semantic gold for paired 2B/4B comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "primary_model_role": "4B sampled historical predictions",
            "paired_model_role": "2B full historical predictions",
            "sample_size": args.sample_size,
            "seed": args.seed,
            "stratification": "balanced by dataset changeflag for sampling only",
            "changeflag_distribution": dict(sorted(flag_counts.items())),
            "annotation_blinding": (
                "Do not show model predictions, references, or changeflag to annotators."
            ),
        },
        "inputs": {
            "annotations": {"path": str(annotations), "sha256": sha256(annotations)},
            "primary_predictions": {"path": str(primary_path), "sha256": sha256(primary_path)},
            "paired_predictions": {"path": str(paired_path), "sha256": sha256(paired_path)},
            "dataset_zip": {"path": str(archive_path), "sha256": sha256(archive_path)},
        },
        "outputs": {
            "image_mapping": "image_mapping.csv",
            "model_pairing_private": "model_pairing_private.csv",
            "image_root": "images",
        },
        "limitations": [
            "Historical predictions have no complete generation_manifest.json.",
            (
                "Any later model score using this gold must be labelled historical exploratory "
                "unless a matching generation manifest is supplied."
            ),
        ],
    }
    (output_dir / "frozen_subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(mapping_rows)} frozen image pairs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
