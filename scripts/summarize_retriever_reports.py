#!/usr/bin/env python3
"""Summarize one or more retriever benchmark JSON reports with bootstrap CIs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def bootstrap(values: list[float], seed: int = 17, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(rounds)]
    samples.sort()
    return samples[int(0.025 * rounds)], samples[int(0.975 * rounds) - 1]


def summarize(paths: list[Path]) -> list[dict[str, object]]:
    output = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("rows", [])
        metrics: dict[str, object] = {}
        for key in ("recall_at_k", "gt_positive_region_coverage", "mean_gt_coverage", "latency_ms"):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            lo, hi = bootstrap(values)
            metrics[key] = {"mean": mean(values) if values else None, "ci95": [lo, hi], "n": len(values)}
        output.append({"provider": report["provider"], "model_id": rows[0].get("model_id") if rows else None,
                       "grid_size": report["grid_size"], "top_k": report["top_k"],
                       "query_mode": report.get("query_mode"), "samples": report["samples"], "metrics": metrics,
                       "source": str(path)})
    output.sort(key=lambda item: float(item["metrics"]["recall_at_k"]["mean"] or -1), reverse=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
