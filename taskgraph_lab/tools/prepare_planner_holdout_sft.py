"""Prepare Planner SFT data with a fixed, externally supplied test set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from taskgraph_lab import PROMPT_VERSION
from taskgraph_lab.taskgraph.dsl import DSL_VERSION
from taskgraph_lab.training.planner_dataset import PlannerSFTDataset, file_sha256

LAB_ROOT = Path(__file__).resolve().parents[1]


def _load_rows(
    paths: list[Path],
    *,
    system_prompt: Path,
    target_format: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        dataset = PlannerSFTDataset(
            path,
            system_prompt=system_prompt,
            target_format=target_format,
        )
        dataset_rows = list(dataset)
        rows.extend(dataset_rows)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "count": len(dataset_rows),
            }
        )
    ids = [str(row["id"]) for row in rows]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate Planner sample ids across inputs: {duplicates[:50]}")
    return rows, sources


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row["metadata"].get(key) or "unknown") for row in rows).items())
    )


def _split_manifest(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "count": len(rows),
        "sample_ids": [str(row["id"]) for row in rows],
        "per_dataset": _counts(rows, "dataset"),
        "per_intent": _counts(rows, "intent"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare TaskGraph Planner SFT data with a fixed held-out test set"
    )
    parser.add_argument("--train-input", type=Path, action="append", required=True)
    parser.add_argument("--test-input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=LAB_ROOT / "prompts" / "planner_student_system_prompt.txt",
    )
    parser.add_argument("--target-format", choices=("dsl", "json"), default="dsl")
    args = parser.parse_args()

    train_rows, train_sources = _load_rows(
        args.train_input,
        system_prompt=args.system_prompt,
        target_format=args.target_format,
    )
    test_rows, test_sources = _load_rows(
        args.test_input,
        system_prompt=args.system_prompt,
        target_format=args.target_format,
    )
    overlap = sorted({str(row["id"]) for row in train_rows} & {str(row["id"]) for row in test_rows})
    if overlap:
        raise ValueError(f"train/test sample-id overlap: {overlap[:50]}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)

    manifest = {
        "version": "taskgraph-planner-heldout-sft-v1",
        "train_sources": train_sources,
        "test_sources": test_sources,
        "system_prompt": str(args.system_prompt.resolve()),
        "system_prompt_sha256": file_sha256(args.system_prompt),
        "prompt_version": PROMPT_VERSION,
        "target_format": args.target_format,
        "planner_dsl_version": DSL_VERSION,
        "split_strategy": "external_disjoint_holdout",
        "population_count": len(train_rows) + len(test_rows),
        "training_population_count": len(train_rows),
        "held_out_test_count": len(test_rows),
        "train_test_overlap_count": 0,
        "splits": {
            "train": _split_manifest(train_path, train_rows),
            "test": _split_manifest(test_path, test_rows),
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "training_population_count": len(train_rows),
        "held_out_test_count": len(test_rows),
        "train_test_overlap_count": 0,
        "train_sha256": manifest["splits"]["train"]["sha256"],
        "test_sha256": manifest["splits"]["test"]["sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
