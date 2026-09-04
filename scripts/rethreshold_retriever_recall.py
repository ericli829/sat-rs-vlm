#!/usr/bin/env python3
"""Recompute grid-retrieval recall from saved rankings at new GT coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from retriever_benchmark import coverage, grid_boxes, sliding_grid_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8, 0.9, 1.0])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("manifest ids must be unique")
    return result


def metric_for_threshold(
    report: dict[str, Any], manifest: dict[str, dict[str, Any]], threshold: float
) -> dict[str, float | int]:
    hits = {1: 0, 3: 0, 5: 0}
    oracle_hits = 0
    total = len(report["rows"])
    for result_row in report["rows"]:
        row_id = str(result_row["id"])
        manifest_row = manifest[row_id]
        gt_boxes = [[float(value) for value in box] for box in manifest_row["gt_boxes"]]
        if len(gt_boxes) != 1:
            raise ValueError(f"{row_id}: expected exactly one GT box")
        with Image.open(manifest_row["image"]) as image:
            window_ratio = report.get("candidate_window_ratio")
            boxes = (
                sliding_grid_boxes(
                    image.width,
                    image.height,
                    int(report["grid_size"]),
                    float(window_ratio),
                )
                if window_ratio is not None
                else grid_boxes(image.width, image.height, int(report["grid_size"]))
            )
        gt = gt_boxes[0]
        positive = {
            index for index, box in enumerate(boxes) if coverage(gt, box) >= threshold
        }
        ranked = [int(index) for index in result_row["ranked_region_indices"]]
        oracle_hits += int(bool(positive))
        for k in hits:
            hits[k] += int(any(index in positive for index in ranked[:k]))
    return {
        "samples": total,
        "recall_at_1": hits[1] / total,
        "recall_at_3": hits[3] / total,
        "recall_at_5": hits[5] / total,
        "oracle_recall": oracle_hits / total,
        "recall_at_1_count": hits[1],
        "recall_at_3_count": hits[3],
        "recall_at_5_count": hits[5],
        "oracle_count": oracle_hits,
    }


def display_name(report: dict[str, Any], path: Path) -> str:
    rows = report.get("rows", [])
    if rows:
        return str(rows[0].get("model_id") or report.get("provider") or path.stem)
    return str(report.get("provider") or path.stem)


def main() -> int:
    args = parse_args()
    thresholds = sorted(set(float(value) for value in args.thresholds))
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be between 0 and 1")
    manifest = load_manifest(args.manifest)
    models = []
    for report_path in args.reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_ids = {str(row["id"]) for row in report["rows"]}
        if report_ids != set(manifest):
            raise ValueError(f"{report_path}: row ids do not match manifest")
        models.append(
            {
                "model": display_name(report, report_path),
                "report": str(report_path.resolve()),
                "thresholds": {
                    str(threshold): metric_for_threshold(report, manifest, threshold)
                    for threshold in thresholds
                },
            }
        )
    payload = {
        "schema_version": 1,
        "definition": "coverage = intersection(candidate, GT) / area(GT)",
        "success": "at least one of the top-k candidate crops has coverage >= threshold",
        "manifest": str(args.manifest.resolve()),
        "thresholds": thresholds,
        "models": models,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VRSBench-200 Recall under stricter GT coverage",
        "",
        "`coverage = intersection(candidate, GT) / area(GT)`. A sample is recalled at k when",
        "at least one of the top-k candidate crops reaches the configured coverage threshold.",
        "",
        "| Coverage threshold | Model | Recall@1 | Recall@3 | Recall@5 | Grid oracle |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        key = str(threshold)
        for model in models:
            metrics = model["thresholds"][key]
            lines.append(
                f"| {threshold:.0%} | {model['model']} | "
                f"{metrics['recall_at_1']:.1%} | {metrics['recall_at_3']:.1%} | "
                f"{metrics['recall_at_5']:.1%} | {metrics['oracle_recall']:.1%} |"
            )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
