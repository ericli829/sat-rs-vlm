"""Qwen3-VL-4B Stage-A v2 的 canonical population 与固定 Stage2 数据构造。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.cyclic_training import (
    load_protected_evaluation_ids,
    sha256_file,
)
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

BUILDER_VERSION = "stage-a-v2-population-1.0"
SOURCE_FILENAMES = {
    "VRSBench": "legal_vrs_train.jsonl",
    "LEVIR-CC": "legal_levir_train.jsonl",
}


def _sample_id(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Canonical population contains a sample without an id")
    return sample_id


def _assert_unique(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    counts = Counter(_sample_id(row) for row in rows)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate sample IDs: {duplicates[:5]}")


def _task_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("task_type", "unknown")) for row in rows).items()))


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_canonical_training_population(
    source_rows: Mapping[str, Sequence[dict[str, Any]]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
    protected_evaluation_manifest: str | Path,
    seed: int,
    source_inputs: Mapping[str, Mapping[str, Any]],
    prompt_profiles: Mapping[str, str],
) -> dict[str, Any]:
    """写出完整合法训练 population，并从实际 tiers manifest 排除所有保护 ID。

    输入必须已经经过正式 source normalization、图片路径改写和 prompt strengthening。
    函数只负责唯一性、评测保护边界、确定性排序、SHA 和 manifest，不重新解释数据。
    """

    missing_sources = sorted(set(SOURCE_FILENAMES).difference(source_rows))
    if missing_sources:
        raise ValueError(f"Canonical population is missing sources: {missing_sources}")
    protected_path = Path(protected_evaluation_manifest)
    protected_ids = load_protected_evaluation_ids(protected_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    populations: dict[str, dict[str, Any]] = {}
    final_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    removed_by_source: dict[str, int] = {}
    all_final_ids: set[str] = set()
    for source in SOURCE_FILENAMES:
        rows = list(source_rows[source])
        _assert_unique(rows, f"{source} source population")
        source_ids = {_sample_id(row) for row in rows}
        cross_source = sorted(all_final_ids.intersection(source_ids))
        if cross_source:
            raise ValueError(
                f"Canonical populations contain cross-source duplicate IDs: {cross_source[:5]}"
            )
        removed = source_ids.intersection(protected_ids)
        legal_rows = [row for row in rows if _sample_id(row) not in protected_ids]
        legal_rows.sort(key=_sample_id)
        legal_ids = {_sample_id(row) for row in legal_rows}
        final_overlap = legal_ids.intersection(protected_ids)
        if final_overlap:
            raise ValueError(
                f"Protected evaluation IDs remain in {source}: {sorted(final_overlap)[:5]}"
            )
        output_path = destination / SOURCE_FILENAMES[source]
        write_jsonl(output_path, legal_rows)
        populations[source] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "sample_count": len(legal_rows),
            "task_distribution": _task_distribution(legal_rows),
            "unique_count": len(legal_ids),
        }
        final_rows_by_source[source] = legal_rows
        removed_by_source[source] = len(removed)
        all_final_ids.update(legal_ids)

    validation = list(validation_rows)
    _assert_unique(validation, "canonical validation")
    validation_ids = {_sample_id(row) for row in validation}
    train_validation_overlap = sorted(all_final_ids.intersection(validation_ids))
    if train_validation_overlap:
        raise ValueError(
            "Canonical train/validation IDs overlap: " + ", ".join(train_validation_overlap[:5])
        )
    validation.sort(key=_sample_id)
    validation_path = destination / "validation.jsonl"
    write_jsonl(validation_path, validation)

    final_overlap_count = len(all_final_ids.intersection(protected_ids))
    if final_overlap_count:
        raise ValueError("Canonical training population still overlaps protected evaluation")
    overlap_removed = sum(removed_by_source.values())
    manifest = {
        "schema_version": "1.0",
        "builder_version": BUILDER_VERSION,
        "seed": int(seed),
        "populations": populations,
        "validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "sample_count": len(validation),
            "task_distribution": _task_distribution(validation),
            "unique_count": len(validation_ids),
        },
        "protected": {
            "manifest": str(protected_path),
            "manifest_sha256": sha256_file(protected_path),
            "ids_count": len(protected_ids),
            "overlap_removed": overlap_removed,
            "overlap_removed_by_source": removed_by_source,
            "final_overlap": final_overlap_count,
        },
        "protected_eval_ids_count": len(protected_ids),
        "protected_overlap_removed": overlap_removed,
        "protected_eval_overlap_count": final_overlap_count,
        "source_inputs": {key: dict(value) for key, value in source_inputs.items()},
        "prompt_profile": dict(prompt_profiles),
    }
    manifest_path = destination / "population_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "population_manifest": str(manifest_path)}


def load_validated_population_manifest(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """读取 population manifest，并校验每个冻结 JSONL 的 SHA、数量和唯一性。"""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("builder_version") != BUILDER_VERSION:
        raise ValueError(
            "Population manifest is not a canonical Stage-A v2 asset: "
            f"expected builder_version={BUILDER_VERSION!r}"
        )
    protected = payload.get("protected")
    if not isinstance(protected, Mapping) or int(protected.get("final_overlap", -1)) != 0:
        raise ValueError("Canonical population manifest must prove protected final_overlap=0")
    records = payload.get("populations", {})
    if not isinstance(records, Mapping):
        raise ValueError(f"Population manifest has no populations mapping: {path}")
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_FILENAMES:
        record = records.get(source)
        if not isinstance(record, Mapping):
            raise ValueError(f"Population manifest is missing source: {source}")
        data_path = Path(str(record.get("path", "")))
        if not data_path.is_absolute() and not data_path.is_file():
            candidate = path.parent / data_path
            data_path = candidate if candidate.is_file() else data_path
        if not data_path.is_file():
            raise FileNotFoundError(f"Population JSONL does not exist: {data_path}")
        actual_sha = sha256_file(data_path)
        if actual_sha != str(record.get("sha256", "")):
            raise ValueError(f"Population SHA mismatch for {source}: {data_path}")
        rows = list(read_jsonl(data_path))
        _assert_unique(rows, f"{source} canonical population")
        if len(rows) != int(record.get("sample_count", -1)):
            raise ValueError(f"Population sample count mismatch for {source}: {data_path}")
        rows_by_source[source] = rows
    return payload, rows_by_source


def build_stage2_vrs_levir_dataset(
    population_manifest: str | Path,
    *,
    output_file: str | Path,
    manifest_file: str | Path,
    seed: int = 42,
    vrs_per_levir: int = 3,
) -> dict[str, Any]:
    """构造 VRS 全覆盖、LEVIR 全覆盖优先且不足时 replay 的固定 Stage2 数据。"""

    if vrs_per_levir < 1:
        raise ValueError("vrs_per_levir must be positive")
    population_payload, rows_by_source = load_validated_population_manifest(population_manifest)
    vrs_rows = list(rows_by_source["VRSBench"])
    levir_rows = list(rows_by_source["LEVIR-CC"])
    if not vrs_rows or not levir_rows:
        raise ValueError("Stage2 requires non-empty VRSBench and LEVIR-CC populations")

    target_levir_exposures = math.ceil(len(vrs_rows) / vrs_per_levir)
    replay_needed = max(0, target_levir_exposures - len(levir_rows))
    replay_order = sorted(levir_rows, key=_sample_id)
    random.Random(_stable_seed(seed, "stage2-levir-replay")).shuffle(replay_order)
    replay_rows: list[dict[str, Any]] = []
    replay_distribution: Counter[str] = Counter()
    existing_ids = {_sample_id(row) for row in [*vrs_rows, *levir_rows]}
    for replay_index in range(replay_needed):
        original = replay_order[replay_index % len(replay_order)]
        original_id = _sample_id(original)
        replay = copy.deepcopy(original)
        exposure_id = f"{original_id}__stage2_replay_{replay_index:06d}"
        if exposure_id in existing_ids:
            raise ValueError(f"Synthetic Stage2 replay ID already exists: {exposure_id}")
        metadata = dict(replay.get("metadata", {}))
        metadata.update(
            {
                "training_source": "LEVIR-CC",
                "cycle_replay": True,
                "stage2_replay": True,
                "replay_original_id": original_id,
                "replay_index": replay_index,
                "replay_reason": "stage2_vrs_levir_3to1_top_up",
            }
        )
        replay["id"] = exposure_id
        replay["metadata"] = metadata
        replay_rows.append(replay)
        replay_distribution[original_id] += 1
        existing_ids.add(exposure_id)

    exposures = [*vrs_rows, *levir_rows, *replay_rows]
    random.Random(_stable_seed(seed, "stage2-exposure-order")).shuffle(exposures)
    _assert_unique(exposures, "Stage2 exposure dataset")
    protected_record = dict(population_payload.get("protected", {}))
    protected_manifest = protected_record.get("manifest")
    if not protected_manifest:
        raise ValueError("Population manifest does not record protected evaluation manifest")
    protected_ids = load_protected_evaluation_ids(str(protected_manifest))
    protected_overlap = sorted({_sample_id(row) for row in exposures}.intersection(protected_ids))
    if protected_overlap:
        raise ValueError(
            f"Stage2 exposures overlap protected evaluation IDs: {protected_overlap[:5]}"
        )

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, exposures)
    levir_exposures = len(levir_rows) + len(replay_rows)
    manifest = {
        "schema_version": "1.0",
        "builder_version": BUILDER_VERSION,
        "seed": int(seed),
        "population_manifest": str(Path(population_manifest)),
        "population_manifest_sha256": sha256_file(population_manifest),
        "train_file": str(destination),
        "sha256": sha256_file(destination),
        "vrs_unique_count": len(vrs_rows),
        "levir_unique_count": len(levir_rows),
        "levir_target_exposures": target_levir_exposures,
        "levir_replay_count": len(replay_rows),
        "total_exposures": len(exposures),
        "source_distribution": {
            "VRSBench": len(vrs_rows),
            "LEVIR-CC": levir_exposures,
        },
        "actual_vrs_per_levir": len(vrs_rows) / max(1, levir_exposures),
        "task_distribution": _task_distribution(exposures),
        "replay_original_distribution": dict(sorted(replay_distribution.items())),
        "unique_exposure_ids": len(exposures),
        "protected_eval_overlap_count": 0,
        "coverage_priority": "retain_all_unique_before_ratio",
    }
    manifest_path = Path(manifest_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
