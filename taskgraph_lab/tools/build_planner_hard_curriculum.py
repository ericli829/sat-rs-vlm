"""Materialize a deterministic hard-topology Planner curriculum from train-only rows."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from taskgraph_lab.training.planner_dataset import file_sha256

HARD_INTENTS = {
    "RELATIONAL_COUNT",
    "OBJECT_RELATION",
    "ROUTE_PLANNING",
    "COMPLEX_REASONING",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _assistant_dsl(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    answers = [message for message in messages if message.get("role") == "assistant"]
    if len(answers) != 1:
        raise ValueError(f"{row.get('id')}: expected exactly one assistant message")
    return str(answers[0].get("content", ""))


def _node_count(row: dict[str, Any]) -> int:
    return sum(
        bool(re.match(r"^n[1-9][0-9]*=", line))
        for line in _assistant_dsl(row).splitlines()
    )


def _node_bucket(count: int) -> str:
    if count >= 10:
        return "10_plus"
    if count >= 7:
        return "7_9"
    if count >= 4:
        return "4_6"
    return "short"


def curriculum_factor(intent: str, node_count: int) -> int:
    """Return sample exposure count for one materialized curriculum epoch."""

    if intent not in HARD_INTENTS:
        return 1
    factor = 2
    if intent == "ROUTE_PLANNING":
        factor += 1
    elif intent == "COMPLEX_REASONING":
        factor += 2
    if 7 <= node_count <= 9:
        factor += 1
    elif node_count >= 10:
        factor += 2
    return factor


def build_curriculum(
    *,
    train_file: Path,
    heldout_file: Path,
    source_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"curriculum output already exists: {output_dir}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = _read_jsonl(train_file)
    heldout_rows = _read_jsonl(heldout_file)
    heldout_ids = {str(row["id"]) for row in heldout_rows}
    source_ids = [str(row["id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source training split contains duplicate ids")
    overlap = sorted(set(source_ids) & heldout_ids)
    if overlap:
        raise ValueError(f"source train/heldout overlap: {overlap[:50]}")

    materialized: list[dict[str, Any]] = []
    factor_distribution: Counter[int] = Counter()
    base_intents: Counter[str] = Counter()
    exposure_intents: Counter[str] = Counter()
    base_buckets: Counter[str] = Counter()
    exposure_buckets: Counter[str] = Counter()
    for row in rows:
        source_id = str(row["id"])
        intent = str((row.get("metadata") or {}).get("intent") or "UNKNOWN")
        nodes = _node_count(row)
        bucket = _node_bucket(nodes)
        factor = curriculum_factor(intent, nodes)
        factor_distribution[factor] += 1
        base_intents[intent] += 1
        exposure_intents[intent] += factor
        base_buckets[bucket] += 1
        exposure_buckets[bucket] += factor
        for repeat_index in range(factor):
            copy = json.loads(json.dumps(row, ensure_ascii=False))
            if repeat_index:
                copy["id"] = f"{source_id}::curriculum_r{repeat_index}"
            metadata = dict(copy.get("metadata") or {})
            metadata.update(
                {
                    "curriculum_source_id": source_id,
                    "curriculum_repeat_index": repeat_index,
                    "curriculum_factor": factor,
                    "curriculum_node_count": nodes,
                    "curriculum_node_bucket": bucket,
                    "curriculum_version": "hard-topology-v1",
                }
            )
            copy["metadata"] = metadata
            materialized.append(copy)

    materialized_ids = [str(row["id"]) for row in materialized]
    if len(materialized_ids) != len(set(materialized_ids)):
        raise AssertionError("materialized curriculum ids are not unique")
    materialized_source_ids = {
        str((row.get("metadata") or {}).get("curriculum_source_id"))
        for row in materialized
    }
    if materialized_source_ids & heldout_ids:
        raise AssertionError("heldout source id leaked into curriculum")

    output_dir.mkdir(parents=True, exist_ok=False)
    train_output = output_dir / "train.jsonl"
    with train_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    test_output = output_dir / "test.jsonl"
    shutil.copyfile(heldout_file, test_output)
    manifest = {
        "version": "taskgraph-planner-hard-topology-curriculum-v1",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_train_file": str(train_file.resolve()),
        "source_train_sha256": file_sha256(train_file),
        "source_train_unique_count": len(rows),
        "heldout_file": str(heldout_file.resolve()),
        "heldout_sha256": file_sha256(heldout_file),
        "heldout_count": len(heldout_rows),
        "train_test_source_overlap_count": 0,
        "curriculum_policy": {
            "easy_factor": 1,
            "hard_base_factor": 2,
            "route_bonus": 1,
            "complex_bonus": 2,
            "node_7_9_bonus": 1,
            "node_10_plus_bonus": 2,
            "hard_intents": sorted(HARD_INTENTS),
        },
        "factor_distribution": dict(sorted(factor_distribution.items())),
        "base_intent_distribution": dict(sorted(base_intents.items())),
        "exposure_intent_distribution": dict(sorted(exposure_intents.items())),
        "base_node_bucket_distribution": dict(sorted(base_buckets.items())),
        "exposure_node_bucket_distribution": dict(sorted(exposure_buckets.items())),
        "population_count": len(materialized) + len(heldout_rows),
        "training_population_count": len(materialized),
        "training_unique_source_count": len(rows),
        "held_out_test_count": len(heldout_rows),
        "target_format": source["target_format"],
        "planner_dsl_version": source["planner_dsl_version"],
        "prompt_version": source["prompt_version"],
        "splits": {
            "train": {
                "path": str(train_output.resolve()),
                "sha256": file_sha256(train_output),
                "count": len(materialized),
                "unique_source_count": len(rows),
            },
            "test": {
                "path": str(test_output.resolve()),
                "sha256": file_sha256(test_output),
                "count": len(heldout_rows),
            },
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--heldout-file", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_curriculum(
        train_file=args.train_file,
        heldout_file=args.heldout_file,
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
