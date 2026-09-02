#!/usr/bin/env python3
"""在 XLRS-Bench-lite 计数样本上跑 benchmark。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counting_system.data.xlrs_lite import load_xlrs_counting
from counting_system.eval.metrics import choice_match, gpu_mem_snapshot, merge_gpu_peak, summarize_counts
from counting_system.executor import CountExecutor
from counting_system.overlay import save_overlay
from counting_system.paths import dataset_root
from counting_system.trace import TraceWriter, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="样本数上限；0 或负数表示全部",
    )
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "xlrs_benchmark"))
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else dataset_root()
    samples = load_xlrs_counting(root, max_samples=args.max_samples)
    if not samples:
        print(f"no counting samples under {root}")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "overlays"
    if not args.no_overlay:
        vis_dir.mkdir(exist_ok=True)
    executor = CountExecutor(
        {
            "detector": {"backend": args.backend},
            "gate": {"enabled": bool(args.gate)},
            "count": {"score_threshold": args.score_threshold, "keep_raw_proposals": False},
        }
    )
    trace = TraceWriter(out)
    rows = []
    pairs = []
    calls: list[int] = []
    latencies: list[float] = []
    backends: set[str] = set()
    gpu_peak = gpu_mem_snapshot()
    t0 = time.perf_counter()
    try:
        for i, sample in enumerate(samples, start=1):
            visual = sample.visual_input()
            result = executor(
                visual,
                sample.target,
                entire=sample.entire,
                score_threshold=args.score_threshold,
                trace=trace,
            )
            pred = result.count
            ref = sample.answer_value
            pairs.append((pred, ref))
            overlay = ""
            if not args.no_overlay:
                overlay = str(
                    save_overlay(
                        sample.image_ref(),
                        result,
                        vis_dir / f"{sample.sample_id}.jpg",
                        title=sample.question[:80],
                    )
                )
            backend = str(result.provenance.get("detector") or args.backend)
            backends.add(backend)
            n_calls = int(result.provenance.get("detector_calls") or 0)
            latency = float(result.provenance.get("latency_sec") or 0.0)
            calls.append(n_calls)
            latencies.append(latency)
            gpu_peak = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "target": sample.target.name,
                    "pred": pred,
                    "ref": ref,
                    "answer_letter": sample.answer_letter,
                    "choice_match": choice_match(pred, sample.options, sample.answer_letter),
                    "entire": sample.entire,
                    "region": sample.region_name,
                    "detector": backend,
                    "detector_calls": n_calls,
                    "latency_sec": latency,
                    "raw_count": result.provenance.get("raw_count"),
                    "fusion": result.provenance.get("fusion"),
                    "overlay": overlay,
                }
            )
            print(
                f"[{i}/{len(samples)}] {sample.sample_id} pred={pred} ref={ref} "
                f"target={sample.target.name} calls={n_calls} backend={backend}",
                flush=True,
            )
    finally:
        executor.close()
    metrics = summarize_counts(pairs)
    metrics["elapsed_sec"] = time.perf_counter() - t0
    metrics["backend"] = sorted(backends)
    metrics["mean_detector_calls"] = (sum(calls) / len(calls)) if calls else 0.0
    metrics["mean_latency_sec"] = (sum(latencies) / len(latencies)) if latencies else 0.0
    metrics["gpu"] = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
    write_json(out / "metrics.json", metrics)
    write_json(out / "predictions.json", rows)
    (out / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
