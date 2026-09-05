"""Counting-merger training population with image-level tier protection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.object_adapter_v0 import (
    canonical_image_identity,
    count_bin,
    extract_answer,
)
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

COUNT_BINS = ("0-2", "3-5", "6-10", "11+")
BUILDER_VERSION = "rs-count-merger-v1.0"


def _sample_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("id", row.get("sample_id", ""))
    return str(value).strip() or f"missing_id_row_{index}"


def _canonical_json_sha(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _protected_population(tier_paths: Sequence[str | Path]) -> tuple[set[str], dict[str, str]]:
    protected: set[str] = set()
    tier_sha: dict[str, str] = {}
    for raw_path in tier_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Protected evaluation tier does not exist: {path}")
        tier_sha[path.name] = file_sha256(path)
        for row in read_jsonl(path):
            identity = canonical_image_identity(row)
            if not identity:
                raise ValueError(f"Protected tier row has no canonical image identity: {path}")
            protected.add(identity)
    return protected, tier_sha


def build_counting_expert_data(
    source_train: str | Path,
    output_dir: str | Path,
    *,
    protected_tiers: Sequence[str | Path],
    source_manifest: str | Path,
) -> dict[str, Any]:
    """Build normal VLM counting rows; parseability is diagnostic only."""

    source = Path(source_train)
    source_manifest_path = Path(source_manifest)
    if not source.is_file():
        raise FileNotFoundError(f"Canonical training JSONL does not exist: {source}")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Canonical source manifest does not exist: {source_manifest_path}")
    rows = list(read_jsonl(source))
    protected_images, protected_tier_sha = _protected_population(protected_tiers)
    counting_rows = [
        row for row in rows if str(row.get("task_type", "")).strip().lower() == "counting"
    ]

    duplicate_counts = Counter(_sample_id(row, index) for index, row in enumerate(counting_rows))
    duplicate_ids = sorted(sample_id for sample_id, count in duplicate_counts.items() if count > 1)
    retained: list[dict[str, Any]] = []
    removed = 0
    answer_parsed = 0
    bin_population = Counter({name: 0 for name in COUNT_BINS})
    unique_images: set[str] = set()
    invalid_images: list[str] = []
    empty_answers: list[str] = []
    non_train_split: list[str] = []
    for index, row in enumerate(counting_rows):
        sample_id = _sample_id(row, index)
        metadata = row.get("metadata", {})
        split = (
            str(metadata.get("split", row.get("split", "train"))).strip().lower()
            if isinstance(metadata, Mapping)
            else "train"
        )
        if split and split != "train":
            non_train_split.append(sample_id)
            continue
        image = canonical_image_identity(row)
        if not image:
            invalid_images.append(sample_id)
            continue
        if image in protected_images:
            removed += 1
            continue
        answer = extract_answer(row)
        if not str(answer).strip():
            empty_answers.append(sample_id)
            continue
        parsed = parse_count(answer)
        if parsed.value is not None:
            answer_parsed += 1
            bin_population[count_bin(parsed.value)] += 1
        unique_images.add(image)
        retained.append(dict(row))

    blockers: list[str] = []
    if invalid_images:
        blockers.append(f"counting rows with invalid image identity: {len(invalid_images)}")
    if empty_answers:
        blockers.append(f"counting rows with empty/damaged answer: {len(empty_answers)}")
    if non_train_split:
        blockers.append(f"non-train rows found in source train file: {len(non_train_split)}")
    if not retained:
        blockers.append("no counting rows remain after image-level protection")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.jsonl"
    write_jsonl(train_path, retained)
    protected_contract = {
        "tier_files": protected_tier_sha,
        "protected_image_count": len(protected_images),
        "protected_images_sha256": _canonical_json_sha(sorted(protected_images)),
    }
    audit = {
        "schema_version": "1.0",
        "builder_version": BUILDER_VERSION,
        "source_rows": len(rows),
        "counting_rows_before_image_exclusion": len(counting_rows),
        "protected_image_count": len(protected_images),
        "rows_removed_by_image_overlap": removed,
        "final_train_rows": len(retained),
        "unique_images": len(unique_images),
        "duplicate_sample_ids": duplicate_ids,
        "duplicate_sample_id_count": len(duplicate_ids),
        "answer_parse_rate": answer_parsed / len(retained) if retained else None,
        "answer_parsed_rows": answer_parsed,
        "count_bin_population": dict(bin_population),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_train_sha256": file_sha256(source),
        "output_train_sha256": file_sha256(train_path),
        "protected_tier_sha256": _canonical_json_sha(protected_contract),
        "protected_tiers": protected_contract,
        "invalid_image_sample_ids": invalid_images[:50],
        "empty_answer_sample_ids": empty_answers[:50],
        "non_train_split_sample_ids": non_train_split[:50],
        "blockers": blockers,
    }
    audit_path = destination / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "builder_version": BUILDER_VERSION,
        "task": "counting",
        "source_train": str(source.resolve()),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": audit["source_manifest_sha256"],
        "train_file": train_path.name,
        "train_sha256": audit["output_train_sha256"],
        "rows": len(retained),
        "protected_tier_sha256": audit["protected_tier_sha256"],
        "image_identity_function": "sat_rs_vlm.data.object_adapter_v0.canonical_image_identity",
        "assistant_target_policy": "original_reference_answer",
        "blockers": blockers,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if blockers:
        raise ValueError("Counting expert data audit blocked: " + "; ".join(blockers))
    return {"train": str(train_path), "manifest": str(manifest_path), "audit": audit}
