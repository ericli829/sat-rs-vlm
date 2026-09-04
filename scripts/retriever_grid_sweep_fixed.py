#!/usr/bin/env python3
"""Evaluate one retriever on several grids with one model load."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sat_rs_vlm.integrations.retrievers.registry import create_retriever_provider  # noqa: E402
from scripts.retriever_benchmark import evaluate_row  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--provider", required=True)
    p.add_argument("--model-path")
    p.add_argument("--checkpoint")
    p.add_argument("--arch", default="ViT-B-32")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--grids", nargs="+", type=int, default=[3, 4, 5])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--query-mode", choices=("original", "category"), default="category")
    p.add_argument("--coverage-threshold", type=float, default=0.5)
    args = p.parse_args()
    config = {k: v for k, v in {"model_path": args.model_path, "checkpoint": args.checkpoint, "arch": args.arch}.items() if v}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()][: args.limit]
    provider = create_retriever_provider(args.provider, config)
    try:
        for grid in args.grids:
            evaluated = [evaluate_row(provider, row, grid, args.top_k, args.coverage_threshold, 0.0, args.query_mode) for row in rows]
            def mean(key):
                vals = [float(r[key]) for r in evaluated if r.get(key) is not None]
                return sum(vals) / len(vals) if vals else None
            report = {
                "schema_version": "region-retriever-benchmark-v1",
                "provider": args.provider, "config": config,
                "grid_size": grid, "top_k": args.top_k, "query_mode": args.query_mode,
                "coverage_threshold": args.coverage_threshold, "gate_threshold": 0.0,
                "samples": len(evaluated),
                "metrics": {k: mean(k) for k in ("recall_at_k", "gt_positive_region_coverage", "mean_gt_coverage", "selected_area_ratio", "latency_ms", "gate_recall", "detector_call_reduction")},
                "rows": evaluated,
            }
            out = args.output_dir / f"{args.provider}_val{args.limit}_g{grid}_k{args.top_k}_{args.query_mode}.json"
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"grid_size": grid, **report["metrics"]}, ensure_ascii=False), flush=True)
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
