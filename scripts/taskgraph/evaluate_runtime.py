"""Run the production TaskGraph runtime over normalized JSONL samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.taskgraph.evaluation_runner import run_taskgraph_evaluation
from sat_rs_vlm.taskgraph.runtime import runtime_from_config


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
    if payload is not None:
        dataset = payload.get("dataset") if isinstance(payload, dict) else None
        rows = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{path} must contain a JSON list or records list")
        return [
            ({"dataset": dataset, **row} if dataset and "dataset" not in row else row)
            for row in rows
        ]
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--stop-on-sample-error", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.provider_config.read_text(encoding="utf-8")) or {}
    runtime = runtime_from_config(config)
    try:
        summary = run_taskgraph_evaluation(
            runtime,
            _rows(args.input),
            args.output,
            continue_on_sample_error=not args.stop_on_sample_error,
            fail_fast=args.fail_fast,
            resume=not args.no_resume,
            image_root=args.image_root,
        )
    finally:
        runtime.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
