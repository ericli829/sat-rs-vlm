#!/usr/bin/env python3
"""Run grid-size/Top-K/query-mode sweeps for one retriever checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.retriever_benchmark import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider", default="git_rsclip")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--arch", default="ViT-B-32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--grid-size", nargs="+", type=int, default=[3, 4])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for grid in args.grid_size:
        for top_k in args.top_k:
            for mode in ("original", "category"):
                provider_config = {
                    "model_path": args.model_path,
                    "checkpoint": args.checkpoint,
                    "arch": args.arch,
                }
                provider_config = {key: value for key, value in provider_config.items() if value}
                report = run_benchmark(
                    args.manifest,
                    args.provider,
                    provider_config,
                    grid_size=grid,
                    top_k=top_k,
                    limit=args.limit,
                    query_mode=mode,
                )
                name = f"{args.provider}_g{grid}_k{top_k}_{mode}.json"
                (args.output_dir / name).write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                summaries.append(
                    {"grid_size": grid, "top_k": top_k, "query_mode": mode, **report["metrics"]}
                )
                print(json.dumps(summaries[-1], ensure_ascii=False))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
