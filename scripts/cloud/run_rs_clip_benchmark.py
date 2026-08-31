#!/usr/bin/env python3
"""Run the five-model RS-CLIP benchmark in explicit, resumable GPU tiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.retriever_benchmark import evaluate_row  # noqa: E402

from sat_rs_vlm.configuration.environment import expand_environment  # noqa: E402
from sat_rs_vlm.integrations.retrievers.registry import (  # noqa: E402
    create_retriever_provider,
)

EXPECTED_MODELS = (
    "remoteclip",
    "georsclip",
    "farslip",
    "satelliteclip",
    "git_rsclip",
)
REPORT_METRICS = (
    "recall_at_k",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "reciprocal_rank",
    "average_precision",
    "ndcg_at_k",
    "oracle_recall",
    "random_recall_at_k",
    "gt_positive_region_coverage",
    "mean_gt_coverage",
    "top1_gt_coverage",
    "topk_union_gt_coverage",
    "mean_selected_roi_area_ratio",
    "selected_union_area_ratio",
    "processed_area_ratio",
    "selected_area_ratio",
    "latency_ms",
    "gate_recall",
    "detector_call_reduction",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_config(path: Path, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("cloud benchmark config must be a mapping")
    expanded = expand_environment(payload, environ=environ or os.environ)
    models = expanded.get("models")
    if not isinstance(models, Mapping) or tuple(models) != EXPECTED_MODELS:
        raise ValueError("models must be ordered exactly as: " + ", ".join(EXPECTED_MODELS))
    return dict(expanded)


def load_manifest(path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            row_id = str(row.get("id", "")).strip()
            if not row_id or row_id in seen_ids:
                raise ValueError(f"manifest id is empty or duplicated at line {line_number}")
            seen_ids.add(row_id)
            if not str(row.get("query", "")).strip():
                raise ValueError(f"manifest row {row_id} has no query")
            if not str(row.get("category", "")).strip():
                raise ValueError(f"manifest row {row_id} has no category query")
            boxes = row.get("gt_boxes")
            if not isinstance(boxes, list) or not boxes:
                raise ValueError(f"manifest row {row_id} has no gt_boxes")
            image = Path(str(row.get("image", ""))).expanduser()
            if not image.is_absolute():
                image = dataset_root / image
            normalized = dict(row)
            normalized["id"] = row_id
            normalized["image"] = str(image.resolve())
            rows.append(normalized)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def stable_stage_rows(
    rows: list[dict[str, Any]], seed: int, limit: int | None
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )
    if limit is not None and len(ordered) < limit:
        raise ValueError(f"tier requires {limit} rows but manifest has only {len(ordered)}")
    return ordered if limit is None else ordered[:limit]


def validate_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique_images: set[str] = set()
    categories: set[str] = set()
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        unique_images.add(str(image_path))
        categories.add(str(row["category"]))
        for index, raw_box in enumerate(row["gt_boxes"]):
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                raise ValueError(f"row {row['id']} GT box {index} is not xyxy")
            box = [float(value) for value in raw_box]
            if not all(math.isfinite(value) for value in box):
                raise ValueError(f"row {row['id']} GT box {index} is not finite")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"row {row['id']} GT box {index} is degenerate")
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise ValueError(
                    f"row {row['id']} GT box {index} is outside {width}x{height}: {box}"
                )
            if max(box) <= 2.0 and min(width, height) > 32:
                raise ValueError(
                    f"row {row['id']} GT box {index} appears normalized, "
                    "but absolute_xyxy is required"
                )
    return {
        "rows": len(rows),
        "unique_images": len(unique_images),
        "categories": len(categories),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def aggregate_rows(rows: list[dict[str, Any]], warmup_rows: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in REPORT_METRICS:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        metrics[key] = statistics.mean(values) if values else None
    latency = [float(row["latency_ms"]) for row in rows[warmup_rows:]]
    metrics["steady_latency_ms"] = {
        "warmup_rows_excluded": min(warmup_rows, len(rows)),
        "mean": statistics.mean(latency) if latency else None,
        "median": statistics.median(latency) if latency else None,
        "p90": percentile(latency, 0.90),
        "p95": percentile(latency, 0.95),
    }
    metrics["cache_hits"] = sum(int(row.get("cache_hits", 0)) for row in rows)
    return metrics


def load_checkpoint_rows(path: Path, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    actual_ids = [str(row["id"]) for row in rows]
    expected_ids = [str(row["id"]) for row in expected[: len(rows)]]
    if actual_ids != expected_ids:
        raise ValueError(f"resume checkpoint does not match staged manifest: {path}")
    return rows


def hardware_info(torch: Any) -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": cuda,
        "torch_cuda": str(torch.version.cuda),
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_vram_mb": properties.total_memory / 1024**2,
            }
        )
    return payload


def model_provider_config(model: Mapping[str, Any], allow_cpu: bool) -> dict[str, Any]:
    ignored = {"display_name", "provider"}
    config = {str(key): value for key, value in model.items() if key not in ignored}
    if allow_cpu:
        config["device"] = "cpu"
    checkpoint = config.get("checkpoint")
    if checkpoint is not None and not Path(str(checkpoint)).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {checkpoint}")
    model_path = config.get("model_path")
    if model_path is not None and not Path(str(model_path)).expanduser().exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    return config


def validate_model_configs(
    models: Mapping[str, Mapping[str, Any]],
    selected_models: list[str],
    allow_cpu: bool,
) -> None:
    """Fail before a run starts when any selected model artifact is missing."""
    for model_name in selected_models:
        model_provider_config(models[model_name], allow_cpu)


def report_is_complete(
    path: Path,
    *,
    expected_samples: int,
    stage_manifest_sha: str,
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "complete"
        and int(payload.get("samples", -1)) == expected_samples
        and payload.get("stage_manifest_sha256") == stage_manifest_sha
    )


def run_model(
    *,
    model_name: str,
    model: Mapping[str, Any],
    rows: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    output_dir: Path,
    source_manifest_sha: str,
    stage_manifest_sha: str,
    tier: str,
    warmup_rows: int,
    allow_cpu: bool,
    torch: Any,
) -> dict[str, Any]:
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / "report.json"
    if report_is_complete(
        report_path,
        expected_samples=len(rows),
        stage_manifest_sha=stage_manifest_sha,
    ):
        print(f"[{tier}] {model_name}: complete, skipping")
        return json.loads(report_path.read_text(encoding="utf-8"))

    checkpoint_path = model_dir / "rows.jsonl"
    completed = load_checkpoint_rows(checkpoint_path, rows)
    provider_config = model_provider_config(model, allow_cpu)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    provider = create_retriever_provider(str(model["provider"]), provider_config)
    started = time.perf_counter()
    try:
        with checkpoint_path.open("a", encoding="utf-8", buffering=1) as handle:
            for index, row in enumerate(rows[len(completed) :], len(completed)):
                result = evaluate_row(
                    provider,
                    row,
                    int(protocol["grid_size"]),
                    int(protocol["top_k"]),
                    float(protocol["coverage_threshold"]),
                    float(protocol.get("gate_threshold", 0.0)),
                    str(protocol["query_mode"]),
                )
                result["sequence_index"] = index
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed.append(result)
                if (index + 1) % 10 == 0 or index + 1 == len(rows):
                    print(f"[{tier}] {model_name}: {index + 1}/{len(rows)}")
    finally:
        provider.close()

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    )
    metrics = aggregate_rows(completed, warmup_rows)
    report = {
        "schema_version": "rs-clip-cloud-benchmark-v1",
        "status": "complete",
        "tier": tier,
        "provider": model["provider"],
        "model_name": model_name,
        "model_id": model.get("model_id", model["display_name"]),
        "display_name": model["display_name"],
        "device": provider_config.get("device"),
        "samples": len(completed),
        "source_manifest_sha256": source_manifest_sha,
        "stage_manifest_sha256": stage_manifest_sha,
        "protocol": dict(protocol),
        "metrics": metrics,
        "peak_vram_mb": peak_vram_mb,
        "elapsed_ms_this_invocation": (time.perf_counter() - started) * 1000.0,
        "rows": completed,
    }
    atomic_json(report_path, report)
    with (model_dir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def verify_prerequisite(
    output_root: Path,
    required_tier: str | None,
    source_manifest_sha: str,
) -> None:
    if required_tier is None:
        return
    status_path = output_root / required_tier / "tier_status.json"
    if not status_path.is_file():
        raise RuntimeError(
            f"tier {required_tier!r} must complete before this tier: {status_path} missing"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("models") != list(EXPECTED_MODELS):
        raise RuntimeError(f"tier {required_tier!r} is not complete for all five models")
    if status.get("source_manifest_sha256") != source_manifest_sha:
        raise RuntimeError("prerequisite tier used a different source manifest")


def write_summary(output_dir: Path, reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking = []
    for report in reports:
        metrics = report["metrics"]
        latency = metrics["steady_latency_ms"]
        ranking.append(
            {
                "model": report["display_name"],
                "provider": report["provider"],
                "samples": report["samples"],
                "recall_at_1": metrics["recall_at_1"],
                "recall_at_3": metrics["recall_at_3"],
                "recall_at_5": metrics["recall_at_5"],
                "mrr": metrics["reciprocal_rank"],
                "ndcg_at_5": metrics["ndcg_at_k"],
                "mean_gt_coverage": metrics["mean_gt_coverage"],
                "selected_area_ratio": metrics["selected_area_ratio"],
                "latency_p50_ms": latency["median"],
                "latency_p95_ms": latency["p95"],
                "peak_vram_mb": report["peak_vram_mb"],
            }
        )
    ranking.sort(
        key=lambda item: (
            -float(item["recall_at_5"] or -1),
            -float(item["ndcg_at_5"] or -1),
            float(item["latency_p50_ms"] or float("inf")),
        )
    )
    atomic_json(output_dir / "ranking.json", ranking)
    with (output_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    lines = [
        f"# RS-CLIP cloud ranking: {reports[0]['tier']}",
        "",
        "| Rank | Model | R@1 | R@3 | R@5 | MRR | NDCG@5 | Coverage | "
        "P50 ms | P95 ms | Peak VRAM MB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(ranking, 1):
        percent = lambda value: "N/A" if value is None else f"{100 * value:.2f}%"  # noqa: E731
        number = lambda value: "N/A" if value is None else f"{value:.2f}"  # noqa: E731
        lines.append(
            f"| {index} | {item['model']} | {percent(item['recall_at_1'])} | "
            f"{percent(item['recall_at_3'])} | {percent(item['recall_at_5'])} | "
            f"{number(item['mrr'])} | {number(item['ndcg_at_5'])} | "
            f"{percent(item['mean_gt_coverage'])} | {number(item['latency_p50_ms'])} | "
            f"{number(item['latency_p95_ms'])} | {number(item['peak_vram_mb'])} |"
        )
    (output_dir / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--only-model", choices=EXPECTED_MODELS)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    tiers = config["tiers"]
    if args.tier not in tiers:
        raise ValueError(f"unknown tier {args.tier!r}; choose one of {', '.join(tiers)}")
    experiment = config["experiment"]
    dataset = config["dataset"]
    protocol = config["protocol"]
    manifest_path = Path(dataset["manifest"]).expanduser().resolve()
    dataset_root = Path(dataset["root"]).expanduser().resolve()
    output_root = Path(experiment["output_root"]).expanduser().resolve()
    source_sha = sha256_file(manifest_path)
    all_rows = load_manifest(manifest_path, dataset_root)
    tier_config = tiers[args.tier]
    stage_rows = stable_stage_rows(
        all_rows,
        int(experiment["seed"]),
        tier_config.get("limit"),
    )
    audit = validate_images(stage_rows)
    print(json.dumps({"tier": args.tier, "audit": audit}, indent=2))
    if args.validate_only:
        return 0

    output_dir = output_root / args.tier
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = output_dir / "manifest.jsonl"
    write_jsonl(stage_manifest, stage_rows)
    stage_sha = sha256_file(stage_manifest)
    verify_prerequisite(output_root, tier_config.get("requires"), source_sha)

    import torch

    hardware = hardware_info(torch)
    if bool(experiment.get("require_cuda", True)) and not args.allow_cpu:
        if not hardware["cuda_available"]:
            raise RuntimeError("CUDA is required; pass --allow-cpu only for local debugging")
    plan = {
        "tier": args.tier,
        "samples_per_model": len(stage_rows),
        "models": [args.only_model] if args.only_model else list(EXPECTED_MODELS),
        "source_manifest_sha256": source_sha,
        "stage_manifest_sha256": stage_sha,
        "hardware": hardware,
        "protocol": protocol,
    }
    atomic_json(output_dir / "run_plan.json", plan)
    selected_models = [args.only_model] if args.only_model else list(EXPECTED_MODELS)
    validate_model_configs(config["models"], selected_models, args.allow_cpu)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    reports = []
    for model_name in selected_models:
        reports.append(
            run_model(
                model_name=model_name,
                model=config["models"][model_name],
                rows=stage_rows,
                protocol=protocol,
                output_dir=output_dir,
                source_manifest_sha=source_sha,
                stage_manifest_sha=stage_sha,
                tier=args.tier,
                warmup_rows=int(experiment.get("latency_warmup_rows", 3)),
                allow_cpu=args.allow_cpu,
                torch=torch,
            )
        )
    if set(selected_models) != set(EXPECTED_MODELS):
        print("Partial model run complete; tier remains incomplete.")
        return 0
    ranking = write_summary(output_dir, reports)
    status = {
        "status": "complete",
        "tier": args.tier,
        "models": list(EXPECTED_MODELS),
        "samples_per_model": len(stage_rows),
        "source_manifest_sha256": source_sha,
        "stage_manifest_sha256": stage_sha,
        "hardware": hardware,
        "winner": ranking[0]["model"],
    }
    atomic_json(output_dir / "tier_status.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
