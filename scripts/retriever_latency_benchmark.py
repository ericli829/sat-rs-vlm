#!/usr/bin/env python3
"""Measure steady-state retriever latency after warm-up in an isolated process."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sat_rs_vlm.integrations.retrievers.registry import create_retriever_provider  # noqa: E402
from scripts.retriever_benchmark import evaluate_row  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        key: value
        for key, value in {
            "model_path": args.model_path,
            "checkpoint": args.checkpoint,
            "arch": args.arch,
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "device": "cpu",
        }.items()
        if value is not None
    }
    with args.manifest.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    needed = args.warmup + args.limit
    if len(rows) < needed:
        raise ValueError(f"manifest needs at least {needed} rows")
    provider = create_retriever_provider(args.provider, config)
    measured = []
    try:
        for row in rows[: args.warmup]:
            evaluate_row(provider, row, args.grid_size, args.top_k, 0.5, 0.0, "category")
        for row in rows[args.warmup : needed]:
            measured.append(
                evaluate_row(provider, row, args.grid_size, args.top_k, 0.5, 0.0, "category")
            )
    finally:
        provider.close()
    latencies = [float(row["latency_ms"]) for row in measured]
    return {
        "provider": args.provider,
        "model_id": measured[0]["model_id"],
        "device": "cpu",
        "warmup_rows": args.warmup,
        "measured_rows": len(measured),
        "grid_size": args.grid_size,
        "top_k": args.top_k,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "rows": [{"id": row["id"], "latency_ms": row["latency_ms"]} for row in measured],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--arch", default="ViT-B-32")
    parser.add_argument("--model-id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["latency_ms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
