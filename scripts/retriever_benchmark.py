#!/usr/bin/env python3
"""Benchmark a score-only region retriever on a JSONL region manifest.

Each row contains ``image``, ``query`` and ``gt_boxes`` (absolute xyxy boxes).
The script deliberately uses identical grid candidates for every provider so
VisRAG, SigLIP and remote-sensing CLIP checkpoints are comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sat_rs_vlm.integrations.retrievers.registry import create_retriever_provider  # noqa: E402


def grid_boxes(width: int, height: int, grid_size: int) -> list[tuple[float, float, float, float]]:
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    return [
        (
            width * x / grid_size,
            height * y / grid_size,
            width * (x + 1) / grid_size,
            height * (y + 1) / grid_size,
        )
        for y in range(grid_size)
        for x in range(grid_size)
    ]


def sliding_grid_boxes(
    width: int,
    height: int,
    grid_size: int,
    window_ratio: float,
) -> list[tuple[float, float, float, float]]:
    """Return uniformly spaced, fixed-size windows including both image edges."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if not 0.0 < window_ratio <= 1.0:
        raise ValueError("window_ratio must be in (0, 1]")
    window_width = width * window_ratio
    window_height = height * window_ratio
    x_step = (width - window_width) / (grid_size - 1) if grid_size > 1 else 0.0
    y_step = (height - window_height) / (grid_size - 1) if grid_size > 1 else 0.0
    return [
        (
            x * x_step,
            y * y_step,
            x * x_step + window_width,
            y * y_step + window_height,
        )
        for y in range(grid_size)
        for x in range(grid_size)
    ]


def _area(box: list[float] | tuple[float, ...]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _intersection(
    left: list[float] | tuple[float, ...],
    right: list[float] | tuple[float, ...],
) -> tuple[float, float, float, float] | None:
    box = (
        max(float(left[0]), float(right[0])),
        max(float(left[1]), float(right[1])),
        min(float(left[2]), float(right[2])),
        min(float(left[3]), float(right[3])),
    )
    return box if box[0] < box[2] and box[1] < box[3] else None


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Compute the exact union area of axis-aligned rectangles."""
    if not boxes:
        return 0.0
    x_edges = sorted({edge for box in boxes for edge in (box[0], box[2])})
    total = 0.0
    for x1, x2 in zip(x_edges, x_edges[1:], strict=False):
        intervals = sorted(
            (box[1], box[3]) for box in boxes if box[0] < x2 and box[2] > x1
        )
        if not intervals:
            continue
        covered_y = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered_y += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered_y += end - start
        total += (x2 - x1) * covered_y
    return total


def coverage(gt: list[float], candidate: tuple[float, ...]) -> float:
    intersection = _intersection(gt, candidate)
    return _area(intersection) / _area(gt) if intersection is not None and _area(gt) else 0.0


def evaluate_row(
    provider: Any,
    row: dict[str, Any],
    grid_size: int,
    top_k: int,
    coverage_threshold: float,
    gate_threshold: float,
    query_mode: str = "original",
    candidate_window_ratio: float | None = None,
) -> dict[str, Any]:
    from PIL import Image

    image = Path(str(row["image"])).expanduser().resolve()
    with Image.open(image) as source:
        width, height = source.size
    boxes = (
        sliding_grid_boxes(width, height, grid_size, candidate_window_ratio)
        if candidate_window_ratio is not None
        else grid_boxes(width, height, grid_size)
    )
    query = (
        str(row.get("category") or row["query"]) if query_mode == "category" else str(row["query"])
    )
    result = provider.score_regions(image, query, boxes)
    order = sorted(range(len(boxes)), key=lambda i: (-result.scores[i], i))
    selected = order[: max(1, min(top_k, len(order)))]
    gt_boxes = [[float(value) for value in box] for box in row.get("gt_boxes", [])]
    covered = [max((coverage(gt, boxes[i]) for i in selected), default=0.0) for gt in gt_boxes]
    positive = [
        i
        for i, box in enumerate(boxes)
        if any(coverage(gt, box) >= coverage_threshold for gt in gt_boxes)
    ]
    positive_set = set(positive)
    ranked_positive = [index in positive_set for index in order]

    def recall_at(k: int) -> float:
        return float(bool(positive_set) and any(ranked_positive[: min(k, len(order))]))

    first_positive_rank = next(
        (rank for rank, is_positive in enumerate(ranked_positive, 1) if is_positive), None
    )
    average_precision = None
    if positive_set:
        hits = 0
        precision_sum = 0.0
        for rank, is_positive in enumerate(ranked_positive, 1):
            if is_positive:
                hits += 1
                precision_sum += hits / rank
        average_precision = precision_sum / len(positive_set)
    ndcg_k = min(top_k, len(order))
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_positive in enumerate(ranked_positive[:ndcg_k], 1)
        if is_positive
    )
    ideal_hits = min(len(positive_set), ndcg_k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    random_recall = 0.0
    if positive_set:
        n = len(boxes)
        k = len(selected)
        misses = n - len(positive_set)
        random_recall = 1.0 - (
            math.comb(misses, k) / math.comb(n, k) if misses >= k else 0.0
        )
    top1_covered = [coverage(gt, boxes[order[0]]) for gt in gt_boxes] if order else []
    selected_boxes = [boxes[index] for index in selected]
    union_covered = []
    for gt in gt_boxes:
        intersections = [
            intersection
            for box in selected_boxes
            if (intersection := _intersection(gt, box)) is not None
        ]
        union_covered.append(_union_area(intersections) / _area(gt) if _area(gt) else 0.0)
    processed_area_ratio = sum(_area(box) for box in selected_boxes) / (width * height)
    gate_selected = [i for i, score in enumerate(result.scores) if score >= gate_threshold]
    return {
        "id": row.get("id", image.name),
        "image": str(image),
        "query": query,
        "query_mode": query_mode,
        "candidate_window_ratio": candidate_window_ratio,
        "provider": result.provider,
        "model_id": result.model_id,
        "recall_at_k": float(
            bool(gt_boxes) and any(value >= coverage_threshold for value in covered)
        ),
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "reciprocal_rank": 1.0 / first_positive_rank if first_positive_rank else 0.0,
        "average_precision": average_precision,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else None,
        "oracle_recall": float(bool(positive_set)),
        "random_recall_at_k": random_recall,
        "gt_positive_region_coverage": (
            sum(value >= coverage_threshold for value in covered) / len(gt_boxes)
            if gt_boxes
            else None
        ),
        "mean_gt_coverage": (sum(covered) / len(covered) if covered else None),
        "top1_gt_coverage": (
            sum(top1_covered) / len(top1_covered) if top1_covered else None
        ),
        "topk_union_gt_coverage": (
            sum(union_covered) / len(union_covered) if union_covered else None
        ),
        "mean_selected_roi_area_ratio": (
            processed_area_ratio / len(selected_boxes) if selected_boxes else 0.0
        ),
        "selected_union_area_ratio": _union_area(selected_boxes) / (width * height),
        "processed_area_ratio": processed_area_ratio,
        "selected_area_ratio": processed_area_ratio,
        "scored_regions": len(boxes),
        "selected_regions": len(selected),
        "positive_regions": len(positive_set),
        "ranked_region_indices": order,
        "region_scores": [float(result.scores[i]) for i in range(len(boxes))],
        "gate_positive_regions": len(positive),
        "gate_recall": (
            sum(i in gate_selected for i in positive) / len(positive) if positive else None
        ),
        "detector_call_reduction": (1.0 - len(gate_selected) / len(boxes)),
        "latency_ms": result.latency_ms,
        "cache_hits": result.metadata.get("score_cache_hits", 0),
        "query_cache_hit": result.metadata.get("query_cache_hit", False),
    }


def run_benchmark(
    manifest: Path,
    provider_name: str,
    provider_config: dict[str, Any],
    *,
    grid_size: int = 3,
    top_k: int = 3,
    coverage_threshold: float = 0.5,
    gate_threshold: float = 0.0,
    limit: int | None = None,
    query_mode: str = "original",
    candidate_window_ratio: float | None = None,
) -> dict[str, Any]:
    provider = create_retriever_provider(provider_name, provider_config)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if limit is not None and len(rows) >= limit:
                    break
                try:
                    row = json.loads(line)
                    rows.append(
                        evaluate_row(
                            provider,
                            row,
                            grid_size,
                            top_k,
                            coverage_threshold,
                            gate_threshold,
                            query_mode,
                            candidate_window_ratio,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"invalid benchmark row {line_number}: {exc}") from exc
    finally:
        provider.close()

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "schema_version": "region-retriever-benchmark-v1",
        "provider": provider_name,
        "config": provider_config,
        "grid_size": grid_size,
        "top_k": top_k,
        "query_mode": query_mode,
        "candidate_window_ratio": candidate_window_ratio,
        "coverage_threshold": coverage_threshold,
        "gate_threshold": gate_threshold,
        "samples": len(rows),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "metrics": {
            "recall_at_k": mean("recall_at_k"),
            "recall_at_1": mean("recall_at_1"),
            "recall_at_3": mean("recall_at_3"),
            "recall_at_5": mean("recall_at_5"),
            "reciprocal_rank": mean("reciprocal_rank"),
            "average_precision": mean("average_precision"),
            "ndcg_at_k": mean("ndcg_at_k"),
            "oracle_recall": mean("oracle_recall"),
            "random_recall_at_k": mean("random_recall_at_k"),
            "gt_positive_region_coverage": mean("gt_positive_region_coverage"),
            "mean_gt_coverage": mean("mean_gt_coverage"),
            "top1_gt_coverage": mean("top1_gt_coverage"),
            "topk_union_gt_coverage": mean("topk_union_gt_coverage"),
            "mean_selected_roi_area_ratio": mean("mean_selected_roi_area_ratio"),
            "selected_union_area_ratio": mean("selected_union_area_ratio"),
            "processed_area_ratio": mean("processed_area_ratio"),
            "selected_area_ratio": mean("selected_area_ratio"),
            "latency_ms": mean("latency_ms"),
            "gate_recall": mean("gate_recall"),
            "detector_call_reduction": mean("detector_call_reduction"),
            "cache_hits": sum(int(row["cache_hits"]) for row in rows),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--arch", default="ViT-B-32")
    parser.add_argument("--model-id")
    parser.add_argument("--cache-dir")
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument(
        "--candidate-window-ratio",
        type=float,
        help="fixed crop width/height as a fraction of the image; enables overlapping windows",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-mode", choices=("original", "category"), default="original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = {
        key: value
        for key, value in {
            "model_path": args.model_path,
            "checkpoint": args.checkpoint,
            "arch": args.arch,
            "model_id": args.model_id,
            "cache_dir": args.cache_dir,
        }.items()
        if value
    }
    report = run_benchmark(
        args.manifest,
        args.provider,
        config,
        grid_size=args.grid_size,
        top_k=args.top_k,
        coverage_threshold=args.coverage_threshold,
        gate_threshold=args.gate_threshold,
        limit=args.limit,
        query_mode=args.query_mode,
        candidate_window_ratio=args.candidate_window_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        rows = report["rows"]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
