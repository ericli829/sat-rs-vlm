#!/usr/bin/env python3
"""Calibrate and audit a high-recall detector gate from retriever scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retriever_benchmark import coverage, grid_boxes  # noqa: E402


def _is_calibration_image(image: str, calibration_fraction: float) -> bool:
    digest = hashlib.sha256(image.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < calibration_fraction


def _positive_indices(
    manifest_row: dict[str, Any], grid_size: int, coverage_threshold: float
) -> set[int]:
    from PIL import Image

    with Image.open(manifest_row["image"]) as source:
        boxes = grid_boxes(source.width, source.height, grid_size)
    return {
        index
        for index, box in enumerate(boxes)
        if any(
            coverage([float(value) for value in gt], box) >= coverage_threshold
            for gt in manifest_row.get("gt_boxes", [])
        )
    }


def select_threshold(positive_scores: list[float], target_recall: float) -> float:
    """Return the highest observed threshold meeting recall on calibration positives."""

    if not positive_scores:
        raise ValueError("cannot calibrate a gate without positive tiles")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    ordered = sorted(float(value) for value in positive_scores)
    allowed_misses = math.floor((1.0 - target_recall) * len(ordered) + 1e-12)
    return ordered[min(allowed_misses, len(ordered) - 1)]


def _metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = 0
    retained_positives = 0
    tiles = 0
    retained_tiles = 0
    for record in records:
        scores = record["scores"]
        positive_indices = record["positive_indices"]
        positives += len(positive_indices)
        retained_positives += sum(scores[index] >= threshold for index in positive_indices)
        tiles += len(scores)
        retained_tiles += sum(score >= threshold for score in scores)
    return {
        "rows": len(records),
        "positive_tiles": positives,
        "threshold": threshold,
        "gate_recall": retained_positives / positives if positives else None,
        "detector_call_reduction": 1.0 - retained_tiles / tiles if tiles else None,
        "retained_tiles": retained_tiles,
        "total_tiles": tiles,
    }


def analyze_gate(
    report_path: Path,
    manifest_path: Path,
    *,
    target_recall: float = 0.99,
    calibration_fraction: float = 0.5,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report_rows = report["rows"]
    if len(report_rows) != len(manifest):
        raise ValueError("report and manifest row counts differ")
    grid_size = int(report["grid_size"])
    coverage_threshold = float(report["coverage_threshold"])
    records: list[dict[str, Any]] = []
    for result_row, manifest_row in zip(report_rows, manifest, strict=True):
        if str(result_row["id"]) != str(manifest_row.get("id")):
            raise ValueError("report and manifest row order/id mismatch")
        records.append(
            {
                "id": result_row["id"],
                "image": str(result_row["image"]),
                "scores": [float(value) for value in result_row["region_scores"]],
                "positive_indices": _positive_indices(
                    manifest_row, grid_size, coverage_threshold
                ),
            }
        )
    calibration = [
        record
        for record in records
        if _is_calibration_image(record["image"], calibration_fraction)
    ]
    test = [record for record in records if record not in calibration]
    calibration_positive_scores = [
        record["scores"][index]
        for record in calibration
        for index in record["positive_indices"]
    ]
    threshold = select_threshold(calibration_positive_scores, target_recall)
    targets = (1.0, 0.995, 0.99, 0.98, 0.95)
    all_positive_scores = [
        record["scores"][index]
        for record in records
        for index in record["positive_indices"]
    ]
    return {
        "schema_version": "region-retriever-gate-analysis-v1",
        "provider": report["provider"],
        "model_id": report_rows[0]["model_id"] if report_rows else None,
        "grid_size": grid_size,
        "coverage_threshold": coverage_threshold,
        "target_gate_recall": target_recall,
        "split": "image-cluster deterministic SHA-256",
        "calibration_fraction": calibration_fraction,
        "calibrated_threshold": threshold,
        "calibration": _metrics(calibration, threshold),
        "test": _metrics(test, threshold),
        "overall": _metrics(records, threshold),
        "in_sample_tradeoff": [
            {
                "target_gate_recall": target,
                **_metrics(records, select_threshold(all_positive_scores, target)),
            }
            for target in targets
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{100.0 * value:.2f}%"

    lines = [
        "# Region Retriever Count gate calibration",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Grid: {payload['grid_size']}x{payload['grid_size']}",
        f"- Positive tile definition: GT coverage >= {payload['coverage_threshold']}",
        f"- Split: {payload['split']} ({payload['calibration_fraction']:.0%} calibration)",
        f"- Calibrated threshold: {payload['calibrated_threshold']:.8f}",
        "",
        "| Split | Rows | Positive tiles | GateRecall | Detector-call reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("calibration", "test", "overall"):
        item = payload[name]
        lines.append(
            f"| {name.title()} | {item['rows']} | {item['positive_tiles']} | "
            f"{percent(item['gate_recall'])} | {percent(item['detector_call_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "The threshold is accepted only if held-out GateRecall reaches the target. "
            "The in-sample trade-off is diagnostic, not an unbiased production estimate.",
            "",
            "## In-sample trade-off",
            "",
            "| Target | Actual GateRecall | Detector-call reduction | Threshold |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in payload["in_sample_tradeoff"]:
        lines.append(
            f"| {percent(item['target_gate_recall'])} | {percent(item['gate_recall'])} | "
            f"{percent(item['detector_call_reduction'])} | {item['threshold']:.8f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze_gate(
        args.report,
        args.manifest,
        target_recall=args.target_recall,
        calibration_fraction=args.calibration_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
