"""多源训练 population 的确定性全覆盖分桶。

该模块只负责样本选择与覆盖证明，不读取图片、不编码 token，也不参与 Trainer。
VRSBench 按任务独立打乱并切片；LEVIR-CC 按图像对保留 caption variant 关系，
从而保证一个 cycle 内每个合法样本恰好出现一次，最后不足 bucket 的样本不会丢弃。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sat_rs_vlm.utils.jsonl import read_jsonl

ImageKey = tuple[str, ...]


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_id(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Cyclic training population contains a sample without an id")
    return sample_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_task_population(
    rows: Sequence[dict[str, Any]],
    bucket_sizes: Mapping[str, int],
    *,
    seed: int,
    cycle_index: int,
) -> list[list[dict[str, Any]]]:
    """按 task 分桶；每个任务的并集等于其完整 population 且交集为空。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("task_type", "unknown")), []).append(row)
    missing = sorted(set(bucket_sizes).difference(grouped))
    if missing:
        raise ValueError(f"Cyclic bucket tasks are missing from source: {missing}")
    unknown = sorted(set(grouped).difference(bucket_sizes))
    if unknown:
        raise ValueError(f"Cyclic bucket sizes are missing for tasks: {unknown}")

    task_chunks: dict[str, list[list[dict[str, Any]]]] = {}
    for task in sorted(grouped):
        size = int(bucket_sizes[task])
        if size < 1:
            raise ValueError(f"Cyclic bucket size must be positive for task {task}")
        shuffled = sorted(grouped[task], key=_sample_id)
        random.Random(_stable_seed(seed, cycle_index, task)).shuffle(shuffled)
        task_chunks[task] = [
            shuffled[start : start + size] for start in range(0, len(shuffled), size)
        ]

    num_rounds = max((len(chunks) for chunks in task_chunks.values()), default=0)
    return [
        [
            row
            for task in sorted(task_chunks)
            for row in (
                task_chunks[task][round_index]
                if round_index < len(task_chunks[task])
                else []
            )
        ]
        for round_index in range(num_rounds)
    ]


def partition_group_variants(
    rows: Sequence[dict[str, Any]],
    *,
    variants_per_round: int,
    seed: int,
    cycle_index: int,
    image_key: Callable[[dict[str, Any]], ImageKey],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """按图像组轮换 variant；不使用 modulo 回卷，也不复制不足的最后一组。"""

    if variants_per_round < 1:
        raise ValueError("training_samples_per_image_group must be positive")
    groups: dict[ImageKey, list[dict[str, Any]]] = {}
    for row in rows:
        key = image_key(row)
        if not key:
            raise ValueError(f"Sample {_sample_id(row)} has no image-group key")
        groups.setdefault(key, []).append(row)

    chunks: dict[ImageKey, list[list[dict[str, Any]]]] = {}
    for key in sorted(groups):
        variants = sorted(groups[key], key=_sample_id)
        random.Random(_stable_seed(seed, cycle_index, *key)).shuffle(variants)
        chunks[key] = [
            variants[start : start + variants_per_round]
            for start in range(0, len(variants), variants_per_round)
        ]
    num_rounds = max((len(group_chunks) for group_chunks in chunks.values()), default=0)
    rounds = [
        [
            row
            for key in sorted(chunks)
            for row in (chunks[key][round_index] if round_index < len(chunks[key]) else [])
        ]
        for round_index in range(num_rounds)
    ]
    return rounds, {
        "unique_image_pair_count": len(groups),
        "caption_variant_count": sum(len(group) for group in groups.values()),
        "variants_per_pair": {
            "min": min((len(group) for group in groups.values()), default=0),
            "max": max((len(group) for group in groups.values()), default=0),
        },
        "per_round_variant_count": [len(round_rows) for round_rows in rounds],
    }


def combine_source_rounds(
    source_rounds: Mapping[str, Sequence[Sequence[dict[str, Any]]]],
    *,
    seed: int,
    cycle_index: int,
) -> list[list[dict[str, Any]]]:
    """按 round 合并各 source；每轮仅改变行顺序，不改变成员关系。"""

    num_rounds = max((len(rounds) for rounds in source_rounds.values()), default=0)
    combined: list[list[dict[str, Any]]] = []
    for round_index in range(num_rounds):
        rows = [
            row
            for source in sorted(source_rounds)
            if round_index < len(source_rounds[source])
            for row in source_rounds[source][round_index]
        ]
        random.Random(_stable_seed(seed, cycle_index, "round", round_index)).shuffle(rows)
        combined.append(rows)
    return combined


def validate_cycle_coverage(
    population: Sequence[dict[str, Any]],
    rounds: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """比较 population 与所有 round，返回可写入 manifest 的覆盖证明。"""

    population_ids = [_sample_id(row) for row in population]
    if len(population_ids) != len(set(population_ids)):
        duplicates = sorted(
            sample_id
            for sample_id, count in Counter(population_ids).items()
            if count > 1
        )
        raise ValueError(f"Training population contains duplicate ids: {duplicates[:5]}")
    exposure_ids = [_sample_id(row) for round_rows in rounds for row in round_rows]
    exposure_counts = Counter(exposure_ids)
    population_set = set(population_ids)
    missing = sorted(population_set.difference(exposure_counts))
    unexpected = sorted(set(exposure_counts).difference(population_set))
    duplicate_ids = sorted(sample_id for sample_id, count in exposure_counts.items() if count > 1)
    return {
        "population_samples": len(population_ids),
        "total_exposures": len(exposure_ids),
        "total_unique_samples": len(exposure_counts),
        "coverage_rate": len(population_set.intersection(exposure_counts))
        / max(1, len(population_set)),
        "duplicate_count": sum(count - 1 for count in exposure_counts.values() if count > 1),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "valid": not missing and not unexpected and not duplicate_ids,
    }


def _ids_from_tier(tier: Mapping[str, Any], manifest_path: Path) -> set[str]:
    for key in ("sample_ids", "ids", "protected_ids"):
        value = tier.get(key)
        if isinstance(value, list):
            return {str(item) for item in value}
    path_value = tier.get("path") or tier.get("file")
    if not path_value:
        return set()
    path = Path(str(path_value))
    candidates = [path] if path.is_absolute() else [manifest_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return {_sample_id(row) for row in read_jsonl(candidate)}
    raise FileNotFoundError(
        f"Protected E3 JSONL referenced by manifest was not found: {path_value}"
    )


def load_protected_e3_ids(manifest_path: str | Path) -> set[str]:
    """读取 Unified Evaluation Tiers v2 的完整 E3 ID 集；无法证明时失败。"""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Protected evaluation manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("protected_evaluation_ids", "evaluation_ids", "all_tier_ids"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return {str(item) for item in value}
    tiers = payload.get("tiers", {})
    if isinstance(tiers, Mapping):
        for name in ("E3", "e3", "E3_full", "e3_full"):
            tier = tiers.get(name)
            if isinstance(tier, Mapping):
                ids = _ids_from_tier(tier, path)
                if ids:
                    return ids
    raise ValueError(f"Cannot resolve protected E3 sample IDs from manifest: {path}")


def assert_no_evaluation_leakage(
    rows: Sequence[dict[str, Any]], protected_ids: set[str]
) -> dict[str, Any]:
    train_ids = {_sample_id(row) for row in rows}
    overlap = sorted(train_ids.intersection(protected_ids))
    if overlap:
        raise ValueError(
            "Training population overlaps protected Unified Evaluation E3 IDs: "
            + ", ".join(overlap[:10])
        )
    return {
        "protected_e3_id_count": len(protected_ids),
        "training_id_count": len(train_ids),
        "overlap_count": 0,
        "valid": True,
    }
