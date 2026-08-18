"""多源训练 population 的确定性全覆盖分桶。

该模块只负责样本选择与覆盖证明，不读取图片、不编码 token，也不参与 Trainer。
VRSBench 按任务独立打乱并切片；LEVIR-CC 按图像对保留 caption variant 关系，
从而保证一个 cycle 内每个合法样本恰好出现一次，最后不足 bucket 的样本不会丢弃。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
    target_rounds: int | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """按图像组轮换 variant，并可确定性地摊到指定轮数。"""

    if variants_per_round < 1:
        raise ValueError("training_samples_per_image_group must be positive")
    if target_rounds is not None and target_rounds < 1:
        raise ValueError("target_rounds must be positive")
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
    natural_rounds = max((len(group_chunks) for group_chunks in chunks.values()), default=0)
    num_rounds = target_rounds or natural_rounds
    if target_rounds is None:
        rounds = [
            [
                row
                for key in sorted(chunks)
                for row in (
                    chunks[key][round_index]
                    if round_index < len(chunks[key])
                    else []
                )
            ]
            for round_index in range(num_rounds)
        ]
    else:
        rounds = [[] for _ in range(num_rounds)]
        assignments: dict[ImageKey, set[int]] = {key: set() for key in chunks}
        records = [
            (key, chunk_index, chunk)
            for key in sorted(chunks)
            for chunk_index, chunk in enumerate(chunks[key])
        ]
        records.sort(
            key=lambda item: _stable_seed(
                seed, cycle_index, "group-chunk", *item[0], item[1]
            )
        )
        for key, chunk_index, chunk in records:
            unused = [index for index in range(num_rounds) if index not in assignments[key]]
            candidates = unused or list(range(num_rounds))
            round_index = min(
                candidates,
                key=lambda index: (
                    len(rounds[index]),
                    _stable_seed(
                        seed,
                        cycle_index,
                        "group-round",
                        *key,
                        chunk_index,
                        index,
                    ),
                ),
            )
            rounds[round_index].extend(chunk)
            assignments[key].add(round_index)
    return rounds, {
        "unique_image_pair_count": len(groups),
        "caption_variant_count": sum(len(group) for group in groups.values()),
        "variants_per_pair": {
            "min": min((len(group) for group in groups.values()), default=0),
            "max": max((len(group) for group in groups.values()), default=0),
        },
        "natural_round_count": natural_rounds,
        "scheduled_round_count": num_rounds,
        "per_round_variant_count": [len(round_rows) for round_rows in rounds],
    }


def top_up_source_to_pattern(
    rounds: Sequence[Sequence[dict[str, Any]]],
    source_population: Sequence[dict[str, Any]],
    source_batch_pattern: Sequence[str],
    *,
    replay_source: str,
    reference_source: str,
    seed: int,
    cycle_index: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """按 batch pattern 比例补足短缺 source，并显式标记重复 exposure。"""

    pattern_counts = Counter(str(source) for source in source_batch_pattern)
    if pattern_counts[replay_source] < 1:
        raise ValueError(f"Replay source is absent from source_batch_pattern: {replay_source}")
    if pattern_counts[reference_source] < 1:
        raise ValueError(
            f"Reference source is absent from source_batch_pattern: {reference_source}"
        )
    if not source_population:
        raise ValueError(f"Replay source population is empty: {replay_source}")

    ordered_population = sorted(source_population, key=_sample_id)
    random.Random(
        _stable_seed(seed, cycle_index, "source-replay", replay_source)
    ).shuffle(ordered_population)
    population_ids = {_sample_id(row) for row in ordered_population}
    cursor = 0
    replay_original_counts: Counter[str] = Counter()
    scheduled: list[list[dict[str, Any]]] = []
    per_round: list[dict[str, Any]] = []

    for round_index, round_rows_value in enumerate(rounds):
        round_rows = list(round_rows_value)
        source_counts = Counter(
            str(dict(row.get("metadata", {})).get("training_source", "unknown"))
            for row in round_rows
        )
        reference_count = source_counts[reference_source]
        target_count = math.ceil(
            reference_count
            * pattern_counts[replay_source]
            / pattern_counts[reference_source]
        )
        replay_needed = max(0, target_count - source_counts[replay_source])
        originals_in_round = {
            str(dict(row.get("metadata", {})).get("replay_original_id", row.get("id", "")))
            for row in round_rows
        }
        unseen_in_round = population_ids.difference(originals_in_round)
        added_rows: list[dict[str, Any]] = []
        while len(added_rows) < replay_needed:
            source_row = ordered_population[cursor % len(ordered_population)]
            cursor += 1
            original_id = _sample_id(source_row)
            if original_id in originals_in_round and unseen_in_round:
                continue
            replay_index = len(added_rows)
            replay_row = copy.deepcopy(source_row)
            metadata = dict(replay_row.get("metadata", {}))
            metadata.update(
                {
                    "cycle_replay": True,
                    "replay_original_id": original_id,
                    "replay_cycle_index": cycle_index,
                    "replay_round_index": round_index,
                    "replay_reason": "source_batch_pattern_top_up",
                }
            )
            replay_row["metadata"] = metadata
            replay_row["id"] = (
                f"{original_id}__replay_c{cycle_index:03d}_"
                f"r{round_index:03d}_{replay_index:06d}"
            )
            added_rows.append(replay_row)
            originals_in_round.add(original_id)
            unseen_in_round.discard(original_id)
            replay_original_counts[original_id] += 1

        round_rows.extend(added_rows)
        random.Random(
            _stable_seed(seed, cycle_index, "scheduled-round", round_index)
        ).shuffle(round_rows)
        final_counts = Counter(
            str(dict(row.get("metadata", {})).get("training_source", "unknown"))
            for row in round_rows
        )
        missing = sorted(set(pattern_counts).difference(final_counts))
        if missing:
            raise ValueError(
                f"Scheduled round {round_index} is missing pattern sources: {missing}"
            )
        scheduled.append(round_rows)
        per_round.append(
            {
                "round_index": round_index,
                "reference_source_count": reference_count,
                "replay_source_count_before": source_counts[replay_source],
                "replay_source_target": target_count,
                "replay_exposures_added": len(added_rows),
                "source_distribution_after": dict(sorted(final_counts.items())),
            }
        )

    scheduled_ids = [_sample_id(row) for rows in scheduled for row in rows]
    if len(scheduled_ids) != len(set(scheduled_ids)):
        raise ValueError("Scheduled cycle contains duplicate exposure ids")
    unknown_originals = sorted(set(replay_original_counts).difference(population_ids))
    if unknown_originals:
        raise ValueError(f"Replay exposures reference unknown samples: {unknown_originals[:5]}")
    return scheduled, {
        "enabled": True,
        "replay_source": replay_source,
        "reference_source": reference_source,
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "replay_exposures_added": sum(replay_original_counts.values()),
        "unique_originals_replayed": len(replay_original_counts),
        "max_replays_per_original": max(replay_original_counts.values(), default=0),
        "per_round": per_round,
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
