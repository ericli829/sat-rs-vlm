"""Canonical training population 上的通用、可审计 balanced probe 抽样。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from sat_rs_vlm.data.cyclic_training import (
    load_protected_evaluation_ids,
    sha256_file,
)
from sat_rs_vlm.data.stage_a_v2 import load_validated_population_manifest
from sat_rs_vlm.utils.jsonl import write_jsonl

QuotaShortfallPolicy = Literal["error", "redistribute"]
DuplicatePolicy = Literal["error", "deduplicate"]


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_id(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Probe population contains a sample without an id")
    return sample_id


def _distribution(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        metadata = row.get("metadata", {})
        source = (
            str(metadata.get("training_source", metadata.get("dataset", "unknown")))
            if isinstance(metadata, Mapping)
            else "unknown"
        )
        values[source][str(row.get("task_type", "unknown"))] += 1
    return {source: dict(sorted(counts.items())) for source, counts in sorted(values.items())}


def _stable_sample(
    rows: Sequence[dict[str, Any]], quota: int, *, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_sample_id)
    random.Random(seed).shuffle(ordered)
    return ordered[:quota]


def _format_shortfall(record: Mapping[str, Any]) -> str:
    return (
        "Probe quota cannot be satisfied: "
        f"source={record['source']} task={record['task']} "
        f"requested={record['requested']} available={record['available']} "
        f"shortfall={record['shortfall']}"
    )


def build_balanced_probe_dataset(
    population_manifest: str | Path,
    *,
    targets: Mapping[str, Mapping[str, int]],
    output_dir: str | Path,
    seed: int = 42,
    quota_shortfall_policy: QuotaShortfallPolicy = "error",
    duplicate_policy: DuplicatePolicy = "error",
    total_samples: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """按 source/task 精确配额抽样，默认在任何 quota 不足时立即失败。

    ``redistribute`` 仅用于显式兼容：先保留每个可满足部分，再从 canonical
    population 的剩余样本补足总量。即使总量被补足，manifest 仍将
    ``quota_satisfied`` 标记为 ``false`` 并记录 redistribution。
    """

    if quota_shortfall_policy not in {"error", "redistribute"}:
        raise ValueError("quota_shortfall_policy must be 'error' or 'redistribute'")
    if duplicate_policy not in {"error", "deduplicate"}:
        raise ValueError("duplicate_policy must be 'error' or 'deduplicate'")
    normalized_targets = {
        str(source): {str(task): int(value) for task, value in task_targets.items()}
        for source, task_targets in targets.items()
    }
    invalid = {
        f"{source}/{task}": value
        for source, task_targets in normalized_targets.items()
        for task, value in task_targets.items()
        if value < 0
    }
    if invalid:
        raise ValueError(f"Probe targets must be non-negative: {invalid}")
    requested_total = sum(
        value for task_targets in normalized_targets.values() for value in task_targets.values()
    )
    if total_samples is not None and int(total_samples) != requested_total:
        raise ValueError(
            f"total_samples={total_samples} does not equal requested quotas={requested_total}"
        )

    population_payload, rows_by_source = load_validated_population_manifest(population_manifest)
    protected_record = population_payload.get("protected", {})
    if not isinstance(protected_record, Mapping) or not protected_record.get("manifest"):
        raise ValueError("Population manifest does not record protected evaluation tiers")
    protected_ids = load_protected_evaluation_ids(str(protected_record["manifest"]))

    by_id: dict[str, dict[str, Any]] = {}
    candidate_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    duplicate_ids: list[str] = []
    for source, rows in sorted(rows_by_source.items()):
        candidate_rows_by_source[source] = []
        for row in rows:
            sample_id = _sample_id(row)
            if sample_id in by_id:
                duplicate_ids.append(sample_id)
                if duplicate_policy == "deduplicate":
                    continue
            by_id[sample_id] = row
            candidate_rows_by_source[source].append(row)
    if duplicate_ids and duplicate_policy == "error":
        raise ValueError(f"Probe population contains duplicate IDs: {sorted(duplicate_ids)[:5]}")

    protected_overlap = sorted(set(by_id).intersection(protected_ids))
    if protected_overlap:
        raise ValueError(
            f"Canonical probe population overlaps protected tiers: {protected_overlap[:5]}"
        )
    available_distribution = _distribution(list(by_id.values()))
    shortfall: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for source in sorted(normalized_targets):
        available_tasks = available_distribution.get(source, {})
        source_rows = candidate_rows_by_source.get(source, [])
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            by_task[str(row.get("task_type", "unknown"))].append(row)
        for task in sorted(normalized_targets[source]):
            requested = normalized_targets[source][task]
            available = int(available_tasks.get(task, 0))
            if available < requested:
                shortfall.append(
                    {
                        "source": source,
                        "task": task,
                        "requested": requested,
                        "available": available,
                        "shortfall": requested - available,
                    }
                )
            taken = _stable_sample(
                by_task.get(task, []),
                min(requested, available),
                seed=_stable_seed(seed, source, task),
            )
            selected.extend(taken)
            selected_ids.update(_sample_id(row) for row in taken)

    if shortfall and quota_shortfall_policy == "error":
        raise ValueError(_format_shortfall(shortfall[0]))

    redistribution: dict[str, Any] = {
        "enabled": quota_shortfall_policy == "redistribute",
        "requested": max(0, requested_total - len(selected)),
        "selected": 0,
        "distribution": {},
    }
    if quota_shortfall_policy == "redistribute" and len(selected) < requested_total:
        remaining = [row for sample_id, row in by_id.items() if sample_id not in selected_ids]
        extra = _stable_sample(
            remaining,
            requested_total - len(selected),
            seed=_stable_seed(seed, "redistribution"),
        )
        selected.extend(extra)
        selected_ids.update(_sample_id(row) for row in extra)
        redistribution["selected"] = len(extra)
        redistribution["distribution"] = _distribution(extra)

    selected = sorted(selected, key=_sample_id)
    random.Random(_stable_seed(seed, "probe-output-order")).shuffle(selected)
    selected_id_list = [_sample_id(row) for row in selected]
    if len(selected_id_list) != len(set(selected_id_list)):
        raise ValueError("Balanced probe sampler produced duplicate exposure IDs")
    selected_distribution = _distribution(selected)
    quota_satisfied = not shortfall

    destination = Path(output_dir)
    train_path = destination / "train.jsonl"
    output_sha: str | None = None
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(train_path, selected)
        output_sha = sha256_file(train_path)

    manifest = {
        "schema_version": "2.0",
        "builder": "balanced_probe_dataset",
        "seed": int(seed),
        "requested_total": requested_total,
        "selected_total": len(selected),
        "requested_distribution": normalized_targets,
        "available_distribution": available_distribution,
        "selected_distribution": selected_distribution,
        "shortfall": shortfall,
        "quota_shortfall_policy": quota_shortfall_policy,
        "quota_satisfied": quota_satisfied,
        "redistribution": redistribution,
        "duplicate_policy": duplicate_policy,
        "unique_count": len(selected_id_list),
        "duplicate_count": len(duplicate_ids),
        "protected_eval_ids_count": len(protected_ids),
        "protected_eval_overlap_count": 0,
        "population_manifest": str(Path(population_manifest)),
        "population_manifest_sha256": sha256_file(population_manifest),
        "output_file": str(train_path) if not dry_run else None,
        "output_sha256": output_sha,
        "sample_ids": selected_id_list,
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Balanced Probe Distribution",
            "",
            "| Source | Task | Requested | Available | Selected | Shortfall |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for source in sorted(normalized_targets):
            for task in sorted(normalized_targets[source]):
                requested = normalized_targets[source][task]
                available = available_distribution.get(source, {}).get(task, 0)
                chosen = selected_distribution.get(source, {}).get(task, 0)
                lines.append(
                    f"| {source} | {task} | {requested} | {available} | "
                    f"{chosen} | {max(0, requested - available)} |"
                )
        (destination / "distribution_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return manifest
