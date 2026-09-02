#!/usr/bin/env python3
"""计划书第 13 节必做实验：source scale / threshold / prompt / 去重 / 融合 / tile / gate。"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counting_system.data.xlrs_lite import load_xlrs_counting
from counting_system.detector.fake import FakeDetector
from counting_system.detector.lae_dino import build_detector
from counting_system.eval.metrics import gpu_mem_snapshot, merge_gpu_peak, summarize_counts
from counting_system.eval.protocol import build_protocol_manifest, inventory_system_models
from counting_system.executor import CountExecutor
from counting_system.fusion import duplicate_rate
from counting_system.synth import Blob, write_blob_image
from counting_system.target import build_target, iter_prompt_variants
from counting_system.trace import write_json


def _blobs() -> list[Blob]:
    return [
        Blob((60, 70, 110, 120)),
        Blob((500, 80, 560, 140)),
        Blob((990, 40, 1050, 100)),
        Blob((1700, 90, 1760, 150)),
        Blob((80, 980, 140, 1040)),
        Blob((900, 900, 980, 980)),
        Blob((1600, 1600, 1680, 1680)),
        Blob((40, 1900, 90, 1950)),  # tiny
    ]


THRESHOLDS = (0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02)
SOURCE_SCALES = (1333, 1024, 896)


def _summarize_result(result) -> dict:
    return {
        "count": result.count,
        "raw": result.provenance.get("raw_count"),
        "calls": result.provenance.get("detector_calls"),
        "fusion": result.provenance.get("fusion"),
        "duplicate_rate": result.provenance.get("duplicate_rate"),
        "tiles_planned": result.provenance.get("tiles_planned"),
        "tiles_run": result.provenance.get("tiles_run"),
        "detector": result.provenance.get("detector"),
        "latency_sec": result.provenance.get("latency_sec"),
    }


def run_case(image, target, extra: dict, detector=None) -> dict:
    executor = CountExecutor(extra, detector=detector or FakeDetector())
    result = executor(image, target, entire=True)
    return _summarize_result(result)


def _score_of(item) -> float:
    if isinstance(item, dict):
        return float(item.get("score") or 0.0)
    return float(getattr(item, "score", 0.0))


def run_xlrs_ablation(*, backend: str, max_samples: int, gate: bool, score_threshold: float, out: Path) -> int:
    samples = load_xlrs_counting(max_samples=max_samples)
    if not samples:
        print("no XLRS counting samples")
        return 2
    detector = build_detector({"detector": {"backend": backend}})
    gpu_peak = gpu_mem_snapshot()
    report: dict = {
        "mode": "xlrs_real_detector",
        "backend_requested": backend,
        "backend_actual": getattr(detector, "impl_name", getattr(detector, "name", backend)),
        "max_samples": len(samples),
        "experiments": {},
    }

    def run_sample(sample, extra: dict):
        extra = {
            "detector": {"backend": backend},
            "count": {"score_threshold": extra.get("count", {}).get("score_threshold", score_threshold), "keep_raw_proposals": extra.get("count", {}).get("keep_raw_proposals", False)},
            **{k: v for k, v in extra.items() if k != "count"},
        }
        executor = CountExecutor(extra, detector=detector)
        result = executor(
            sample.visual_input(),
            sample.target,
            entire=sample.entire,
            score_threshold=extra["count"]["score_threshold"],
        )
        rec = _summarize_result(result)
        rec.update(
            {
                "sample_id": sample.sample_id,
                "pred": result.count,
                "ref": sample.answer_value,
                "target": sample.target.name,
                "raw_proposals": result.provenance.get("raw_proposals") if extra["count"].get("keep_raw_proposals") else None,
            }
        )
        return rec, result

    # default run: threshold sweep from raw proposals + 1024 source-scale baseline
    default_rows = []
    threshold_rows: dict[float, list] = {thr: [] for thr in THRESHOLDS}
    try:
        for i, sample in enumerate(samples, start=1):
            rec, result = run_sample(
                sample,
                {
                    "count": {"score_threshold": score_threshold, "keep_raw_proposals": True},
                    "scale": {"default_source_scale": 1024},
                },
            )
            rec.pop("raw_proposals", None)
            rec["source_scale"] = 1024
            default_rows.append({k: v for k, v in rec.items() if k != "raw_proposals"})
            raw = result.provenance.get("raw_proposals") or []
            for thr in THRESHOLDS:
                kept = [d for d in raw if _score_of(d) >= thr]
                threshold_rows[thr].append(
                    {
                        "sample_id": sample.sample_id,
                        "threshold": thr,
                        "raw": len(raw),
                        "after_threshold": len(kept),
                        "dropped_by_threshold": len(raw) - len(kept),
                        "no_proposal": len(raw) == 0,
                        "pred_at_thr": len(kept),
                        "ref": sample.answer_value,
                    }
                )
            gpu_peak = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
            print(f"[default {i}/{len(samples)}] {sample.sample_id} pred={rec['pred']} ref={rec['ref']} calls={rec['calls']}", flush=True)

        report["experiments"]["default_1024"] = {
            "rows": default_rows,
            "summary": summarize_counts([(r["pred"], r["ref"]) for r in default_rows]),
        }
        report["experiments"]["threshold_sweep"] = []
        for thr in THRESHOLDS:
            rows = threshold_rows[thr]
            report["experiments"]["threshold_sweep"].append(
                {
                    "threshold": thr,
                    "mean_after_threshold": sum(r["after_threshold"] for r in rows) / len(rows),
                    "no_proposal_rate": sum(1 for r in rows if r["no_proposal"]) / len(rows),
                    "summary": summarize_counts([(r["pred_at_thr"], r["ref"]) for r in rows]),
                    "rows": rows,
                }
            )

        scale_block = []
        for scale in SOURCE_SCALES:
            if scale == 1024:
                scale_block.append(
                    {
                        "source_scale": scale,
                        "reused_default": True,
                        "summary": report["experiments"]["default_1024"]["summary"],
                        "mean_calls": sum(r["calls"] or 0 for r in default_rows) / len(default_rows),
                    }
                )
                continue
            rows = []
            for i, sample in enumerate(samples, start=1):
                rec, _result = run_sample(sample, {"scale": {"default_source_scale": scale}})
                rec["source_scale"] = scale
                rows.append(rec)
                gpu_peak = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
                print(f"[scale {scale} {i}/{len(samples)}] {sample.sample_id} pred={rec['pred']} ref={rec['ref']}", flush=True)
            scale_block.append(
                {
                    "source_scale": scale,
                    "reused_default": False,
                    "summary": summarize_counts([(r["pred"], r["ref"]) for r in rows]),
                    "mean_calls": sum(r["calls"] or 0 for r in rows) / len(rows),
                    "rows": rows,
                }
            )
        report["experiments"]["source_scale"] = scale_block

        whole_vs_tiled = []
        for mode, extra in (
            (
                "global_only",
                {"scale": {"global": {"enabled": True}, "native": {"enabled": False}, "fine": {"enabled": False}}},
            ),
            (
                "tiled_native",
                {"scale": {"global": {"enabled": False}, "native": {"enabled": True}, "fine": {"enabled": False}}},
            ),
        ):
            rows = []
            for i, sample in enumerate(samples, start=1):
                rec, _result = run_sample(sample, extra)
                rec["mode"] = mode
                rows.append(rec)
                gpu_peak = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
                print(f"[{mode} {i}/{len(samples)}] {sample.sample_id} pred={rec['pred']} ref={rec['ref']}", flush=True)
            whole_vs_tiled.append(
                {
                    "mode": mode,
                    "summary": summarize_counts([(r["pred"], r["ref"]) for r in rows]),
                    "mean_calls": sum(r["calls"] or 0 for r in rows) / len(rows),
                    "rows": rows,
                }
            )
        report["experiments"]["whole_vs_tiled"] = whole_vs_tiled

        if gate:
            rows = []
            for i, sample in enumerate(samples, start=1):
                rec, _result = run_sample(sample, {"gate": {"enabled": True}})
                rec["gate"] = True
                rows.append(rec)
                gpu_peak = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
                print(f"[gate {i}/{len(samples)}] {sample.sample_id} pred={rec['pred']} ref={rec['ref']}", flush=True)
            report["experiments"]["gate"] = {
                "summary": summarize_counts([(r["pred"], r["ref"]) for r in rows]),
                "mean_calls": sum(r["calls"] or 0 for r in rows) / len(rows),
                "rows": rows,
            }
    finally:
        close = getattr(detector, "close", None)
        if close:
            close()

    report["gpu"] = merge_gpu_peak(gpu_peak, gpu_mem_snapshot())
    report["protocol"] = build_protocol_manifest(official_aligned=False)
    report["resources"] = inventory_system_models(
        backends_used=[report["backend_actual"]],
        gate_enabled=bool(gate),
    )
    write_json(out / "report.json", report)
    print(json.dumps({"backend": report["backend_actual"], "default": report["experiments"]["default_1024"]["summary"], "gpu": report["gpu"]}, indent=2))
    print(f"wrote {out / 'report.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "outputs" / "experiments"))
    parser.add_argument("--backend", default="fake", help="fake | auto | grounding_dino")
    parser.add_argument("--max-samples", type=int, default=0, help="真实检测器用 XLRS 样本数；0 且 fake 时跑合成图")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.backend != "fake":
        n = args.max_samples if args.max_samples > 0 else 8
        return run_xlrs_ablation(
            backend=args.backend,
            max_samples=n,
            gate=bool(args.gate),
            score_threshold=args.score_threshold,
            out=out,
        )
    image = write_blob_image(str(out / "synth.png"), width=2048, height=2048, blobs=_blobs())
    expected = len(_blobs())
    report: dict = {"expected": expected, "experiments": {}}

    # 1. source scale
    scale_rows = []
    for scale in (1333, 1024, 896):
        rec = run_case(image, "ship", {"scale": {"default_source_scale": scale}})
        rec["source_scale"] = scale
        scale_rows.append(rec)
    report["experiments"]["source_scale"] = scale_rows

    # 2. threshold sweep（raw 必须保留，截断发生在后处理）
    sweep = []
    base = CountExecutor({"count": {"score_threshold": 0.0, "keep_raw_proposals": True}}, detector=FakeDetector())
    raw_result = base(image, "ship", entire=True)
    raw = raw_result.provenance.get("raw_proposals") or []
    for thr in (0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02):
        kept = [d for d in raw if d["score"] >= thr]
        sweep.append(
            {
                "threshold": thr,
                "raw": len(raw),
                "after_threshold": len(kept),
                "dropped_by_threshold": len(raw) - len(kept),
                "no_proposal": len(raw) == 0,
            }
        )
    report["experiments"]["threshold_sweep"] = sweep

    # 3. prompt wording
    target = build_target("ship")
    prompt_rows = []
    for text in iter_prompt_variants(target):
        rec = run_case(image, build_target("ship", prompt=text), {})
        rec["prompt"] = text
        prompt_rows.append(rec)
    report["experiments"]["prompt_wording"] = prompt_rows

    # 4-5. same-scale dedup / cross-scale fusion ablation
    ablations = []
    for same, cross, native, glob, fine in product(
        [True],
        [True, False],
        [True, False],
        [True, False],
        [True, False],
    ):
        extra = {
            "scale": {
                "global": {"enabled": glob},
                "native": {"enabled": native, "tile_size": 1024, "overlap": 256},
                "fine": {"enabled": fine, "only_for_tiny": False},
            }
        }
        rec = run_case(image, "ship", extra)
        rec.update({"global": glob, "native": native, "fine": fine, "cross": cross})
        ablations.append(rec)
    report["experiments"]["scale_ablation"] = ablations

    # 6. whole-image vs tiled
    report["experiments"]["whole_vs_tiled"] = [
        {
            "mode": "global_only",
            **run_case(
                image,
                "ship",
                {"scale": {"global": {"enabled": True}, "native": {"enabled": False}, "fine": {"enabled": False}}},
            ),
        },
        {
            "mode": "tiled_native",
            **run_case(
                image,
                "ship",
                {"scale": {"global": {"enabled": False}, "native": {"enabled": True}, "fine": {"enabled": False}}},
            ),
        },
    ]

    # 7. optional retriever gate（fake retriever 全 1.0，只验证开关）
    report["experiments"]["gate"] = [
        run_case(image, "ship", {"gate": {"enabled": False}}),
        run_case(image, "ship", {"gate": {"enabled": True, "backend": "fake", "threshold": 0.5}}),
    ]

    # 8. tiny-object 单独统计：面积 < 80^2
    tiny_pred = raw_result.count
    report["experiments"]["tiny_object"] = {
        "tiny_blobs": 1,
        "pred_count": tiny_pred,
        "duplicate_rate": duplicate_rate(list(raw_result.detections)),
    }

    pairs = [(scale_rows[0]["count"], expected)]
    report["summary"] = summarize_counts(pairs)
    write_json(out / "report.json", report)
    print(json.dumps({"expected": expected, "source_scale": scale_rows, "threshold_sweep": sweep}, indent=2, default=str)[:2000])
    print(f"wrote {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
