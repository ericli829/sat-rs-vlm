"""从合法 evaluation JSONL 构建冻结的 E1/E2/E3 评测层级。

该脚本只做数据组织，不加载模型。它按 dataset、task_type 以及任务相关
子类型做确定性轮转采样，保证 E1 是 E2 的子集、E2 是 E3 的子集。若仓库
中已经存在固定 E1，应在配置中将其作为 ``existing_e1_file`` 传入，以保留
历史 593 条评测集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.tier_builder import build_unified_evaluation_tiers

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"Invalid evaluation row at {path}:{line_number}")
            rows.append(row)
    return rows


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _bbox_area(row: dict[str, Any]) -> float | None:
    metadata = _metadata(row)
    value = metadata.get("bbox_clipped", metadata.get("bbox_raw"))
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_bucket(row: dict[str, Any], thresholds: dict[str, float]) -> str:
    area = _bbox_area(row)
    if area is None:
        return "unknown"
    if area < float(thresholds["small_max"]):
        return "small"
    if area < float(thresholds["medium_max"]):
        return "medium"
    return "large"


def _count_value(row: dict[str, Any]) -> int | None:
    reference = row.get("reference")
    if reference is None:
        messages = row.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    reference = message.get("content")
                    break
    match = re.search(r"(?<![\d.])(\d+)(?![\d.])", str(reference or ""))
    return int(match.group(1)) if match else None


def _count_bucket(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 10:
        return "10+"
    if value >= 5:
        return "5-9"
    return str(value)


def _stratum(row: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, ...]:
    metadata = _metadata(row)
    dataset = str(metadata.get("dataset", "unknown"))
    task = str(row.get("task_type", "unknown"))
    if task == "detection":
        subtype = _bbox_bucket(row, thresholds)
    elif task == "counting":
        subtype = _count_bucket(_count_value(row))
    elif task == "vqa":
        subtype = str(metadata.get("qa_type", "unknown"))
    elif task == "change_detection":
        subtype = str(metadata.get("changeflag", "unknown"))
    else:
        subtype = str(metadata.get("source_task", "default"))
    return dataset, task, subtype


def _round_robin_select(
    rows: list[dict[str, Any]],
    target: int,
    *,
    seed: int,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    if target >= len(rows):
        return list(rows)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_stratum(row, thresholds)].append(row)
    rng = random.Random(seed)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda item: str(item["id"]))
        rng.shuffle(bucket_rows)
    keys = sorted(buckets)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < target and keys:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop())
        if all(not values for values in buckets.values()):
            break
        cursor += 1
    return selected


def _distribution(rows: Iterable[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    dataset = Counter()
    task = Counter()
    subtype = Counter()
    for row in rows:
        key = _stratum(row, thresholds)
        dataset[key[0]] += 1
        task[key[1]] += 1
        subtype["/".join(key)] += 1
    return {
        "sample_count": sum(dataset.values()),
        "dataset": dict(sorted(dataset.items())),
        "task": dict(sorted(task.items())),
        "subtype": dict(sorted(subtype.items())),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if str(payload.get("schema_version", "1.0")) == "2.0":
        return build_unified_evaluation_tiers(config_path, project_root=PROJECT_ROOT)
    seed = int(payload.get("seed", 42))
    data_cfg = dict(payload.get("data", {}))
    source_files = [_project_path(value) for value in data_cfg.get("source_files", [])]
    if not source_files:
        raise ValueError("evaluation tier config requires data.source_files")
    rows: list[dict[str, Any]] = []
    for path in source_files:
        rows.extend(_read_jsonl(path))
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Evaluation source files contain duplicate sample IDs")
    train_files = [_project_path(value) for value in data_cfg.get("train_files", [])]
    train_ids = {
        str(row["id"])
        for path in train_files
        for row in _read_jsonl(path)
    }
    overlap = sorted(set(by_id) & train_ids)
    if overlap:
        raise ValueError(f"Evaluation/training ID leakage detected; first IDs: {overlap[:5]}")
    thresholds = dict(
        payload.get("stratification", {}).get(
            "detection_area_thresholds",
            {"small_max": 0.01, "medium_max": 0.10},
        )
    )
    tier_cfg = dict(payload.get("tiers", {}))
    existing_e1 = tier_cfg.get("E1", {}).get("existing_e1_file")
    if existing_e1:
        e1_rows = _read_jsonl(_project_path(existing_e1))
        missing = [str(row["id"]) for row in e1_rows if str(row["id"]) not in by_id]
        if missing:
            raise ValueError(f"Existing E1 contains IDs absent from population: {missing[:5]}")
        e1_ids = {str(row["id"]) for row in e1_rows}
        e1_rows = [by_id[str(row["id"])] for row in e1_rows]
    else:
        e1_rows = _round_robin_select(
            rows,
            min(int(tier_cfg.get("E1", {}).get("target_samples", 593)), len(rows)),
            seed=seed,
            thresholds=thresholds,
        )
    e1_ids = {str(row["id"]) for row in e1_rows}
    e2_target = min(int(tier_cfg.get("E2", {}).get("target_samples", 3000)), len(rows))
    remaining = [row for row in rows if str(row["id"]) not in e1_ids]
    e2_rows = e1_rows + _round_robin_select(
        remaining,
        max(0, e2_target - len(e1_rows)),
        seed=seed + 1,
        thresholds=thresholds,
    )
    e2_ids = {str(row["id"]) for row in e2_rows}
    e3_rows = list(rows)
    output_dir = _project_path(payload.get("output_dir", "data/evaluation/tiers"))
    tier_rows = {"E1": e1_rows, "E2": e2_rows, "E3": e3_rows}
    tier_paths = {
        "E1": output_dir / "e1_quick.jsonl",
        "E2": output_dir / "e2_standard.jsonl",
        "E3": output_dir / "e3_full.jsonl",
    }
    for tier, tier_data in tier_rows.items():
        _write_jsonl(tier_paths[tier], tier_data)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "source_files": [str(path) for path in source_files],
        "train_files": [str(path) for path in train_files],
        "excluded_train_ids": len(overlap),
        "stratification": {"detection_area_thresholds": thresholds},
        "population_distribution": _distribution(rows, thresholds),
        "tiers": {},
        "E1_subset_of_E2": e1_ids <= e2_ids,
        "E2_subset_of_E3": e2_ids <= {str(row["id"]) for row in e3_rows},
    }
    for tier, tier_data in tier_rows.items():
        manifest["tiers"][tier] = {
            "path": str(tier_paths[tier]),
            "sample_count": len(tier_data),
            "sha256": _sha256(tier_paths[tier]),
            "distribution": _distribution(tier_data, thresholds),
            "sample_ids": [str(row["id"]) for row in tier_data],
        }
    manifest_path = output_dir / "evaluation_tiers_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/evaluation_tiers.yaml"))
    args = parser.parse_args()
    manifest = build(args.config.resolve())
    print(
        json.dumps(
            {"output_dir": str(_project_path(manifest["tiers"]["E2"]["path"]).parent), "tiers": manifest["tiers"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
