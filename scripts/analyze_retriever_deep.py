#!/usr/bin/env python3
"""Deep, paired analysis of retriever reports on a shared manifest."""

# Generated Markdown tables intentionally contain long lines.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "reciprocal_rank",
    "average_precision",
    "ndcg_at_k",
    "mean_gt_coverage",
    "top1_gt_coverage",
    "topk_union_gt_coverage",
    "random_recall_at_k",
    "oracle_recall",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _size_bin(ratio: float) -> str:
    if ratio < 0.005:
        return "tiny(<0.5%)"
    if ratio < 0.02:
        return "small(0.5-2%)"
    if ratio < 0.10:
        return "medium(2-10%)"
    return "large(>=10%)"


def _mcnemar_exact(a: list[float], b: list[float]) -> dict[str, float | int]:
    a_only = sum(x > y for x, y in zip(a, b, strict=True))
    b_only = sum(y > x for x, y in zip(a, b, strict=True))
    n = a_only + b_only
    if not n:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _cluster_bootstrap_delta(
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    row_to_image: dict[str, str],
    key: str,
    *,
    seed: int = 17,
    rounds: int = 4000,
) -> tuple[float, float]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_id, image in row_to_image.items():
        grouped[image].append(row_id)
    images = sorted(grouped)
    rng = random.Random(seed)
    deltas = []
    for _ in range(rounds):
        sampled = rng.choices(images, k=len(images))
        values = []
        for image in sampled:
            for row_id in grouped[image]:
                av = a_rows[row_id].get(key)
                bv = b_rows[row_id].get(key)
                if av is not None and bv is not None:
                    values.append(float(av) - float(bv))
        deltas.append(mean(values))
    deltas.sort()
    return deltas[int(0.025 * rounds)], deltas[int(0.975 * rounds) - 1]


def _cluster_permutation_p(
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    row_to_image: dict[str, str],
    key: str,
    *,
    seed: int = 23,
    rounds: int = 20000,
) -> float:
    cluster_sums: dict[str, float] = defaultdict(float)
    for row_id, image in row_to_image.items():
        cluster_sums[image] += float(a_rows[row_id][key]) - float(b_rows[row_id][key])
    values = list(cluster_sums.values())
    observed = abs(sum(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(rounds):
        permuted = abs(sum(value if rng.random() < 0.5 else -value for value in values))
        extreme += permuted >= observed
    return (extreme + 1) / (rounds + 1)


def analyze(
    manifest_path: Path, report_paths: list[Path], old_manifest: Path | None = None
) -> dict[str, Any]:
    manifest = _load_jsonl(manifest_path)
    manifest_by_id = {str(row["id"]): row for row in manifest}
    row_to_image = {row_id: str(row["image"]) for row_id, row in manifest_by_id.items()}
    dimensions: dict[str, tuple[int, int]] = {}
    enriched: dict[str, dict[str, Any]] = {}
    for row_id, row in manifest_by_id.items():
        image_path = str(row["image"])
        if image_path not in dimensions:
            with Image.open(image_path) as image:
                dimensions[image_path] = image.size
        width, height = dimensions[image_path]
        box = row["gt_boxes"][0]
        area_ratio = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (width * height)
        enriched[row_id] = {**row, "area_ratio": area_ratio, "size_bin": _size_bin(area_ratio)}

    reports = []
    expected_ids = list(manifest_by_id)
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["rows"]
        ids = [str(row["id"]) for row in rows]
        if ids != expected_ids:
            raise ValueError(f"report rows do not match manifest order: {path}")
        reports.append(
            (str(rows[0]["model_id"]), report, {str(row["id"]): row for row in rows}, path)
        )

    model_summaries = []
    for model, _report, by_id, path in reports:
        rows = list(by_id.values())
        # Older in-flight reports represented infeasible-grid random recall as
        # null. Its unconditional expectation is zero for those rows.
        for row in rows:
            if row.get("random_recall_at_k") is None and not row.get("positive_regions"):
                row["random_recall_at_k"] = 0.0
        metrics = {key: _mean(rows, key) for key in METRICS}
        random_recall = metrics["random_recall_at_k"]
        oracle = metrics["oracle_recall"]
        measured = metrics["recall_at_5"]
        metrics["normalized_gain_over_random"] = (
            (measured - random_recall) / (oracle - random_recall)
            if measured is not None
            and random_recall is not None
            and oracle is not None
            and oracle > random_recall
            else None
        )
        by_size = {}
        for size in ("tiny(<0.5%)", "small(0.5-2%)", "medium(2-10%)", "large(>=10%)"):
            subset = [
                by_id[row_id] for row_id, info in enriched.items() if info["size_bin"] == size
            ]
            by_size[size] = {
                "n": len(subset),
                "recall_at_1": _mean(subset, "recall_at_1"),
                "recall_at_3": _mean(subset, "recall_at_3"),
                "recall_at_5": _mean(subset, "recall_at_5"),
                "mean_gt_coverage": _mean(subset, "mean_gt_coverage"),
                "random_recall_at_k": _mean(subset, "random_recall_at_k"),
                "oracle_recall": _mean(subset, "oracle_recall"),
            }
        by_category = {}
        category_counts = Counter(str(row.get("category") or "unknown") for row in manifest)
        for category, count in sorted(category_counts.items()):
            if count < 5:
                continue
            subset = [
                by_id[row_id]
                for row_id, info in enriched.items()
                if str(info.get("category") or "unknown") == category
            ]
            by_category[category] = {
                "n": count,
                "recall_at_5": _mean(subset, "recall_at_5"),
                "mean_gt_coverage": _mean(subset, "mean_gt_coverage"),
            }
        model_summaries.append(
            {
                "model": model,
                "source": str(path),
                "metrics": metrics,
                "by_size": by_size,
                "by_category": by_category,
            }
        )
    model_summaries.sort(key=lambda item: float(item["metrics"]["recall_at_5"] or -1), reverse=True)

    pairwise = []
    for (a_model, _, a_rows, _), (b_model, _, b_rows, _) in itertools.combinations(reports, 2):
        a_recall = [float(a_rows[row_id]["recall_at_5"]) for row_id in expected_ids]
        b_recall = [float(b_rows[row_id]["recall_at_5"]) for row_id in expected_ids]
        mcnemar = _mcnemar_exact(a_recall, b_recall)
        coverage_delta = [
            float(a_rows[row_id]["mean_gt_coverage"]) - float(b_rows[row_id]["mean_gt_coverage"])
            for row_id in expected_ids
        ]
        recall_ci = _cluster_bootstrap_delta(a_rows, b_rows, row_to_image, "recall_at_5")
        coverage_ci = _cluster_bootstrap_delta(a_rows, b_rows, row_to_image, "mean_gt_coverage")
        cluster_p = _cluster_permutation_p(a_rows, b_rows, row_to_image, "recall_at_5")
        pairwise.append(
            {
                "model_a": a_model,
                "model_b": b_model,
                "recall_delta_a_minus_b": mean(a_recall) - mean(b_recall),
                "recall_delta_cluster_ci95": list(recall_ci),
                "coverage_delta_a_minus_b": mean(coverage_delta),
                "coverage_delta_cluster_ci95": list(coverage_ci),
                "coverage_wins_ties_losses": {
                    "wins": sum(value > 0 for value in coverage_delta),
                    "ties": sum(value == 0 for value in coverage_delta),
                    "losses": sum(value < 0 for value in coverage_delta),
                },
                "mcnemar": mcnemar,
                "cluster_permutation_p": cluster_p,
            }
        )
    adjusted = _holm([float(item["cluster_permutation_p"]) for item in pairwise])
    for item, value in zip(pairwise, adjusted, strict=True):
        item["cluster_permutation_holm_p"] = value

    old_changed = None
    if old_manifest:
        old_by_id = {str(row["id"]): row for row in _load_jsonl(old_manifest)}
        old_changed = sum(
            old_by_id[row_id].get("gt_boxes") != row.get("gt_boxes")
            for row_id, row in manifest_by_id.items()
        )
    size_counts = Counter(info["size_bin"] for info in enriched.values())
    image_counts = Counter(row_to_image.values())
    return {
        "protocol": {
            "manifest": str(manifest_path),
            "rows": len(manifest),
            "grid_size": 3,
            "top_k": 5,
            "query_mode": "category",
            "coverage_threshold": 0.5,
        },
        "dataset_audit": {
            "unique_images": len(image_counts),
            "rows_on_repeated_images": sum(count for count in image_counts.values() if count > 1),
            "max_rows_per_image": max(image_counts.values()),
            "categories": len(set(str(row.get("category")) for row in manifest)),
            "size_counts": dict(size_counts),
            "boxes_changed_from_old_manifest": old_changed,
        },
        "models": model_summaries,
        "pairwise": pairwise,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(result: dict[str, Any]) -> str:
    audit = result["dataset_audit"]
    lines = [
        "# Deep RS-CLIP evaluation on corrected VRSBench-200",
        "",
        "All models use the same corrected 200 annotations, 3x3 grid, Top-5, category query, coverage threshold 0.5, and CPU.",
        "",
        "## Dataset audit",
        "",
        f"- 200 annotations from {audit['unique_images']} unique images; {audit['rows_on_repeated_images']} rows belong to repeated images (maximum {audit['max_rows_per_image']} rows/image).",
        f"- {audit['boxes_changed_from_old_manifest']} GT boxes changed after fixing normalized-coordinate scaling and clipping.",
        f"- Size distribution: {', '.join(f'{key}: {value}' for key, value in audit['size_counts'].items())}.",
        "- Confidence intervals use image-cluster bootstrap; annotations from the same image are resampled together.",
        "",
        "## Overall ranking",
        "",
        "| Rank | Model | R@1 | R@3 | R@5 | MRR | AP | NDCG@5 | Mean coverage | Random R@5 | Oracle R@5 | Normalized gain |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(result["models"], 1):
        m = item["metrics"]
        lines.append(
            f"| {rank} | {item['model']} | {_pct(m['recall_at_1'])} | {_pct(m['recall_at_3'])} | {_pct(m['recall_at_5'])} | {m['reciprocal_rank']:.3f} | {m['average_precision']:.3f} | {m['ndcg_at_k']:.3f} | {_pct(m['mean_gt_coverage'])} | {_pct(m['random_recall_at_k'])} | {_pct(m['oracle_recall'])} | {m['normalized_gain_over_random']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recall@5 by target size",
            "",
            "| Model | Tiny | Small | Medium | Large |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in result["models"]:
        s = item["by_size"]
        lines.append(
            f"| {item['model']} | {_pct(s['tiny(<0.5%)']['recall_at_5'])} | {_pct(s['small(0.5-2%)']['recall_at_5'])} | {_pct(s['medium(2-10%)']['recall_at_5'])} | {_pct(s['large(>=10%)']['recall_at_5'])} |"
        )
    baseline = result["models"][0]["by_size"]
    lines.append(
        f"| Random Top-5 | {_pct(baseline['tiny(<0.5%)']['random_recall_at_k'])} | {_pct(baseline['small(0.5-2%)']['random_recall_at_k'])} | {_pct(baseline['medium(2-10%)']['random_recall_at_k'])} | {_pct(baseline['large(>=10%)']['random_recall_at_k'])} |"
    )
    lines.append(
        f"| Oracle grid limit | {_pct(baseline['tiny(<0.5%)']['oracle_recall'])} | {_pct(baseline['small(0.5-2%)']['oracle_recall'])} | {_pct(baseline['medium(2-10%)']['oracle_recall'])} | {_pct(baseline['large(>=10%)']['oracle_recall'])} |"
    )
    lines.extend(
        [
            "",
            "## Pairwise inference",
            "",
            "Recall deltas use image-cluster bootstrap. Image-cluster permutation p-values are adjusted across all ten model pairs with Holm's method; row-level exact McNemar results remain in the JSON audit.",
            "",
            "| A vs B | R@5 delta | Cluster 95% CI | Coverage W/T/L | Cluster Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in result["pairwise"]:
        lo, hi = item["recall_delta_cluster_ci95"]
        wtl = item["coverage_wins_ties_losses"]
        lines.append(
            f"| {item['model_a']} vs {item['model_b']} | {100 * item['recall_delta_a_minus_b']:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] pp | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} | {item['cluster_permutation_holm_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Compare model R@5 against the random and Oracle baselines, not against zero.",
            "- Overlapping pairwise cluster intervals or Holm-adjusted p >= 0.05 do not support a statistically reliable superiority claim.",
            "- Parallel CPU runs are suitable for quality comparison but not strict latency ranking; latency requires isolated warm-up runs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.reports, args.old_manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["dataset_audit"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
