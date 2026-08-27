#!/usr/bin/env python3
"""Run a small, staged UHR Locator diagnostic sweep with auditable rendering."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.locators.config import load_locator_config  # noqa: E402
from sat_rs_vlm.integrations.locators.geometry import (  # noqa: E402
    bbox_coverage,
    clamp_bbox,
)
from sat_rs_vlm.integrations.locators.registry import create_locator  # noqa: E402
from sat_rs_vlm.integrations.locators.types import BBox, LocatorError  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LocatorError(f"{label} does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LocatorError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LocatorError(f"{label} must be a mapping")
    payload["_source_path"] = str(resolved)
    payload["_source_sha256"] = _sha256(resolved)
    return payload


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    payload = _load_yaml(path, label="UHR locator experiment config")
    base_overrides = payload.get("base_overrides", {})
    if not isinstance(base_overrides, Mapping):
        raise LocatorError("experiment base_overrides must be a mapping")
    presets = payload.get("presets")
    if not isinstance(presets, Mapping) or not presets:
        raise LocatorError("experiment config requires a non-empty presets mapping")
    for name, preset in presets.items():
        if not str(name).strip() or not isinstance(preset, Mapping):
            raise LocatorError("each experiment preset must be a named mapping")
        overrides = preset.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise LocatorError(f"preset {name!r} overrides must be a mapping")
    return payload


def load_sample_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_yaml(path, label="UHR locator sample manifest")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise LocatorError("sample manifest requires a non-empty samples list")
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise LocatorError("each sample must be a mapping")
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id or sample_id in seen:
            raise LocatorError("sample ids must be non-empty and unique")
        seen.add(sample_id)
        if not str(sample.get("image", "")).strip():
            raise LocatorError(f"sample {sample_id!r} has no image")
        if not str(sample.get("question", "")).strip():
            raise LocatorError(f"sample {sample_id!r} has no question")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_environment_path(value: Any, *, label: str) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise LocatorError(f"{label} contains an unresolved environment variable: {expanded}")
    path = Path(expanded).expanduser().resolve()
    if not path.is_file():
        raise LocatorError(f"{label} does not exist: {path}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preset", action="append")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--detector-provider")
    parser.add_argument("--retriever-provider")
    detector = parser.add_mutually_exclusive_group()
    detector.add_argument("--enable-detector", action="store_true")
    detector.add_argument("--disable-detector", action="store_true")
    retriever = parser.add_mutually_exclusive_group()
    retriever.add_argument("--enable-retriever", action="store_true")
    retriever.add_argument("--disable-retriever", action="store_true")
    return parser.parse_args()


def _apply_provider_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    detector = config.setdefault("detector", {})
    retriever = config.setdefault("retriever", {})
    scorers = config.setdefault("scorers", {})
    detector_scorer = scorers.setdefault("detector", {})
    retrieval_scorer = scorers.setdefault("retrieval", {})
    if args.detector_provider:
        detector["provider"] = args.detector_provider
        detector["enabled"] = True
        detector_scorer["enabled"] = True
    if args.retriever_provider:
        retriever["provider"] = args.retriever_provider
        retriever["enabled"] = True
        retrieval_scorer["enabled"] = True
    if args.enable_detector:
        detector["enabled"] = True
        detector_scorer["enabled"] = True
    if args.disable_detector:
        detector["enabled"] = False
        detector_scorer["enabled"] = False
    if args.enable_retriever:
        retriever["enabled"] = True
        retrieval_scorer["enabled"] = True
    if args.disable_retriever:
        retriever["enabled"] = False
        retrieval_scorer["enabled"] = False


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=max(size, 10))
    except TypeError:
        return ImageFont.load_default()


def _box(values: Sequence[Any]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise LocatorError(f"bbox must contain four values, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _center(box: Sequence[Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = _box(box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _draw_search_overlay(
    image_path: Path,
    trace: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    depth: int | None,
    selected_only: bool = False,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 500))
    font = _font(max(11, round(min(image.size) / 70)))
    by_id = {str(item["region_id"]): item for item in trace}
    root_center = (image.width / 2.0, image.height / 2.0)
    items = [item for item in trace if depth is None or int(item["depth"]) == depth]
    if selected_only:
        items = [item for item in items if bool(item.get("selected"))]
    for item in items:
        selected = bool(item.get("selected"))
        core = _box(item["core_xyxy"])
        parent = by_id.get(str(item.get("parent_id")))
        parent_center = _center(parent["core_xyxy"]) if parent is not None else root_center
        color = (30, 235, 90) if selected else (130, 130, 130)
        draw.line((parent_center, _center(core)), fill=(70, 120, 220), width=width)
        draw.rectangle(core, outline=color, width=width)
        probability = float(item.get("selection_probability", 0.0))
        reasons = ",".join(str(value) for value in item.get("stop_reasons", [])) or "continue"
        label = (
            f"{item['region_id']} d{item['depth']} s={float(item['fused_score']):.2f} "
            f"p={probability:.2f} {reasons}"
        )
        draw.text((core[0] + width, core[1] + width), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_final_overlay(
    image_path: Path,
    regions: Sequence[Sequence[Any]],
    scores: Sequence[Any],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, round(min(image.size) / 350))
    font = _font(max(12, round(min(image.size) / 55)))
    for index, (box, score) in enumerate(zip(regions, scores, strict=True)):
        xyxy = _box(box)
        draw.rectangle(xyxy, outline=(255, 40, 40), width=width)
        draw.text(
            (xyxy[0] + width, xyxy[1] + width),
            f"ROI {index + 1} score={float(score):.3f}",
            fill=(255, 40, 40),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _detector_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    detector = payload.get("provider_provenance", {}).get("detector")
    if not isinstance(detector, Mapping):
        return {"raw": [], "deduplicated": [], "tiles": [], "raw_count": 0, "dedup_count": 0}
    metadata = detector.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    deduplicated: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    raw_count = 0
    for query in metadata.get("queries", []):
        if not isinstance(query, Mapping):
            continue
        provider_metadata = query.get("metadata", {})
        if not isinstance(provider_metadata, Mapping):
            provider_metadata = {}
        target_phrase = str(query.get("target_phrase", ""))
        deduplicated.extend(
            {"box": box, "score": score, "query": target_phrase}
            for box, score in zip(
                query.get("boxes_xyxy", []),
                query.get("scores", []),
                strict=False,
            )
        )
        query_raw = provider_metadata.get("raw_proposals", [])
        if isinstance(query_raw, list):
            raw.extend(
                {**item, "query": target_phrase}
                for item in query_raw
                if isinstance(item, dict)
            )
        query_tiles = provider_metadata.get("tiles", [])
        if isinstance(query_tiles, list):
            tiles.extend(
                {**item, "query": target_phrase}
                for item in query_tiles
                if isinstance(item, dict)
            )
        raw_count += int(
            provider_metadata.get("raw_proposal_count", query.get("proposal_count", 0))
        )
    if not deduplicated:
        deduplicated = [
            {"box": box, "score": score, "query": ""}
            for box, score in zip(
                metadata.get("boxes_xyxy", []),
                metadata.get("scores", []),
                strict=False,
            )
        ]
    if not raw:
        raw = [
            {
                "global_box_xyxy": item["box"],
                "score": item["score"],
                "query": item.get("query", ""),
            }
            for item in deduplicated
        ]
    return {
        "raw": raw,
        "deduplicated": deduplicated,
        "tiles": tiles,
        "raw_count": raw_count or len(raw),
        "dedup_count": len(deduplicated),
    }


def _draw_detector_overlay(
    image_path: Path,
    detector: Mapping[str, Any],
    output_path: Path,
    query: str,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 500))
    font = _font(max(11, round(min(image.size) / 70)))
    for tile in detector.get("tiles", []):
        draw.rectangle(_box(tile["tile_xyxy"]), outline=(30, 180, 220), width=width)
    for item in detector.get("raw", []):
        box = item.get("global_box_xyxy", item.get("box"))
        if box:
            draw.rectangle(_box(box), outline=(230, 190, 30), width=width)
    for item in detector.get("deduplicated", []):
        box = _box(item["box"])
        draw.rectangle(box, outline=(255, 40, 40), width=width * 2)
        draw.text(
            (box[0], box[1]),
            f"{str(item.get('query', ''))[:24]} {float(item['score']):.2f}".strip(),
            fill=(255, 40, 40),
            font=font,
        )
    banner = f"query={query}"
    text_bbox = draw.textbbox((0, 0), banner, font=font)
    banner_height = max(25, text_bbox[3] - text_bbox[1] + 8)
    draw.rectangle((0, 0, image.width, banner_height), fill=(0, 0, 0))
    draw.text((4, 4), banner, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _export_crops(
    image_path: Path,
    regions: Sequence[Sequence[Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        for index, box in enumerate(regions):
            path = output_dir / f"roi_{index + 1:02d}.png"
            image.crop(tuple(round(value) for value in _box(box))).save(path)
            paths.append(path)
    return paths


def _contact_sheet(
    final_overlay_path: Path,
    crop_paths: Sequence[Path],
    payload: Mapping[str, Any],
    sample: Mapping[str, Any],
    output_path: Path,
) -> None:
    canvas = Image.new("RGB", (1500, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    body_font = _font(17)
    draw.text((30, 20), f"Question: {sample['question']}", fill="black", font=title_font)
    draw.text(
        (30, 58),
        f"Reference: {sample.get('reference_answer', 'unknown')}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (30, 88),
        (
            f"processed={float(payload['processed_area_ratio']):.3f}  "
            f"selected_union={float(payload['selected_union_area_ratio']):.3f}  "
            f"depth={payload['depth_reached']}  final_rois={len(payload['regions_xyxy'])}"
        ),
        fill="black",
        font=body_font,
    )
    with Image.open(final_overlay_path) as source:
        overview = source.convert("RGB")
        overview.thumbnail((700, 820))
    canvas.paste(overview, (30, 145))
    details = list(payload.get("region_details", []))
    for index, crop_path in enumerate(crop_paths[:6]):
        row, column = divmod(index, 2)
        x = 780 + column * 350
        y = 145 + row * 285
        with Image.open(crop_path) as source:
            crop = source.convert("RGB")
            crop.thumbnail((320, 210))
        canvas.paste(crop, (x, y))
        detail = details[index] if index < len(details) else {}
        components = detail.get("score_components", {})
        component_text = ", ".join(
            f"{name}={float(value.get('raw', 0.0)):.2f}"
            for name, value in components.items()
            if name in {"detector", "retrieval", "spatial"} and isinstance(value, Mapping)
        )
        draw.text(
            (x, y + 215),
            (
                f"ROI {index + 1} depth={detail.get('depth', '?')} "
                f"score={float(detail.get('score', 0.0)):.3f}"
            ),
            fill="black",
            font=body_font,
        )
        draw.text((x, y + 240), component_text, fill="black", font=body_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_sample_artifacts(
    image_path: Path,
    payload: Mapping[str, Any],
    sample: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    trace = payload.get("search_trace", [])
    depths = sorted({int(item["depth"]) for item in trace})
    overlay_paths: list[str] = []
    for depth in depths:
        path = output_dir / f"search_depth_{depth}.png"
        _draw_search_overlay(image_path, trace, path, depth=depth)
        overlay_paths.append(str(path))
    summary_overlay = output_dir / "search_final.png"
    _draw_search_overlay(image_path, trace, summary_overlay, depth=None, selected_only=True)
    final_overlay = output_dir / "final_roi_overlay.png"
    _draw_final_overlay(image_path, payload["regions_xyxy"], payload["scores"], final_overlay)
    detector = _detector_payload(payload)
    detector_overlay = output_dir / "detector_proposals.png"
    _draw_detector_overlay(image_path, detector, detector_overlay, str(sample["question"]))
    crops = _export_crops(image_path, payload["regions_xyxy"], output_dir / "crops")
    contact_sheet = output_dir / "contact_sheet.png"
    _contact_sheet(final_overlay, crops, payload, sample, contact_sheet)
    return {
        "depth_overlays": overlay_paths,
        "search_final": str(summary_overlay),
        "final_roi_overlay": str(final_overlay),
        "detector_proposals": str(detector_overlay),
        "crops": [str(path) for path in crops],
        "contact_sheet": str(contact_sheet),
        "detector": detector,
    }


def _absolute_gt_boxes(
    sample: Mapping[str, Any],
    width: int,
    height: int,
) -> list[BBox]:
    boxes: list[BBox] = []
    for item in sample.get("gt_bboxes", []):
        if not isinstance(item, Mapping) or "bbox" not in item:
            continue
        values = [float(value) for value in item["bbox"]]
        if item.get("coordinate_mode", "absolute") == "normalized":
            values = [
                values[0] * width,
                values[1] * height,
                values[2] * width,
                values[3] * height,
            ]
        try:
            boxes.append(clamp_bbox(values, width, height))
        except LocatorError:
            continue
    return boxes


def _gt_metrics(
    final_regions: Sequence[Sequence[Any]],
    proposals: Sequence[Mapping[str, Any]],
    gt_boxes: Sequence[BBox],
) -> dict[str, Any]:
    if not gt_boxes:
        return {
            "target_coverage": None,
            "coverage_at_k": None,
            "gt_center_covered": None,
            "gt_bbox_containment_ratio": None,
            "object_coverage": None,
            "proposal_recall": None,
        }
    rois = [_box(region) for region in final_regions]
    proposal_boxes = [_box(item["box"]) for item in proposals if item.get("box")]
    per_gt = [
        max((bbox_coverage(gt, roi) for roi in rois), default=0.0) for gt in gt_boxes
    ]
    center_covered = []
    for gt in gt_boxes:
        center = _center(gt)
        center_covered.append(
            any(roi[0] <= center[0] <= roi[2] and roi[1] <= center[1] <= roi[3] for roi in rois)
        )
    coverage_at_k = {
        str(k): statistics.fmean(
            max((bbox_coverage(gt, roi) for roi in rois[:k]), default=0.0)
            for gt in gt_boxes
        )
        for k in sorted({1, min(3, len(rois)), len(rois)})
        if k > 0
    }
    proposal_recall = statistics.fmean(
        max((bbox_coverage(gt, proposal) for proposal in proposal_boxes), default=0.0)
        >= 0.5
        for gt in gt_boxes
    ) if proposal_boxes else 0.0
    return {
        "target_coverage": statistics.fmean(per_gt),
        "coverage_at_k": coverage_at_k,
        "gt_center_covered": statistics.fmean(center_covered),
        "gt_bbox_containment_ratio": statistics.fmean(per_gt),
        "object_coverage": statistics.fmean(value >= 0.5 for value in per_gt),
        "proposal_recall": proposal_recall,
    }


def collect_metrics(
    payload: Mapping[str, Any],
    sample: Mapping[str, Any],
    image_size: tuple[int, int],
    detector: Mapping[str, Any],
) -> dict[str, Any]:
    trace = list(payload.get("search_trace", []))
    retrieval_scores = [
        float(item["score_components"]["retrieval"]["raw"])
        for item in trace
        if item.get("score_components", {}).get("retrieval", {}).get("available")
    ]
    depth_entries: dict[int, Mapping[str, Any]] = {}
    for item in trace:
        depth_entries.setdefault(int(item["depth"]), item)
    beam_selected_k = {
        str(depth): int(item.get("depth_selected_count", 0))
        for depth, item in sorted(depth_entries.items())
    }
    beam_entropy = {
        str(depth): float(item.get("beam_entropy", 0.0))
        for depth, item in sorted(depth_entries.items())
    }
    gt = _gt_metrics(
        payload.get("regions_xyxy", []),
        detector.get("deduplicated", []),
        _absolute_gt_boxes(sample, *image_size),
    )
    return {
        "depth_reached": int(payload["depth_reached"]),
        "evaluated_region_count": len(trace),
        "selected_region_count": sum(bool(item.get("selected")) for item in trace),
        "processed_area_ratio": float(payload["processed_area_ratio"]),
        "selected_union_area_ratio": float(payload["selected_union_area_ratio"]),
        "processed_union_area_ratio": float(payload["processed_union_area_ratio"]),
        "detector_latency_ms": float(payload["latency_ms"].get("detector", 0.0)),
        "retrieval_latency_ms": float(payload["latency_ms"].get("retrieval", 0.0)),
        "search_latency_ms": float(payload["latency_ms"].get("search", 0.0)),
        "total_latency_ms": float(payload["latency_ms"].get("total", 0.0)),
        "lae_raw_proposal_count": int(detector.get("raw_count", 0)),
        "lae_deduplicated_proposal_count": int(detector.get("dedup_count", 0)),
        "mean_retrieval_score": (
            statistics.fmean(retrieval_scores) if retrieval_scores else None
        ),
        "max_retrieval_score": max(retrieval_scores) if retrieval_scores else None,
        "beam_entropy_by_depth": beam_entropy,
        "beam_selected_k_by_depth": beam_selected_k,
        "final_roi_count": len(payload.get("regions_xyxy", [])),
        "dedup_count": int(detector.get("dedup_count", 0)),
        **gt,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_summary(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in fieldnames} for row in rows
        )
    review_fields = [
        "preset",
        "sample_id",
        "target_covered",
        "multi_region_coverage",
        "context_sufficient",
        "over_search",
        "under_search",
        "best_preset",
        "notes",
    ]
    with (output_root / "human_review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(
            {"preset": row["preset"], "sample_id": row["sample_id"]} for row in rows
        )

    numeric = [
        "processed_area_ratio",
        "selected_union_area_ratio",
        "total_latency_ms",
        "depth_reached",
        "final_roi_count",
        "target_coverage",
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["preset"]), []).append(row)
    lines = [
        "# UHR Locator diagnostic summary",
        "",
        "> Diagnostic development output only; not formal benchmark tuning.",
        "",
        "## Preset aggregates",
        "",
        (
            "| Preset | Samples | Processed area | Selected union | "
            "Latency ms | Depth | ROIs | GT coverage |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for preset, preset_rows in grouped.items():
        means: dict[str, float | None] = {}
        for key in numeric:
            values = [float(row[key]) for row in preset_rows if row.get(key) is not None]
            means[key] = statistics.fmean(values) if values else None
        formatted = [
            preset,
            str(len(preset_rows)),
            f"{means['processed_area_ratio']:.3f}",
            f"{means['selected_union_area_ratio']:.3f}",
            f"{means['total_latency_ms']:.1f}",
            f"{means['depth_reached']:.2f}",
            f"{means['final_roi_count']:.2f}",
            (
                f"{means['target_coverage']:.3f}"
                if means["target_coverage"] is not None
                else "n/a"
            ),
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    lines.extend(["", "## Per-sample artifacts", ""])
    for row in rows:
        relative = Path(str(row["artifact_dir"])).relative_to(output_root)
        lines.extend(
            [
                f"### {row['preset']} / {row['sample_id']}",
                "",
                f"- [contact sheet]({relative.as_posix()}/contact_sheet.png)",
                f"- [search trace]({relative.as_posix()}/search_trace.json)",
                f"- [result]({relative.as_posix()}/result.json)",
                "",
            ]
        )
    lines.extend(
        [
            "## Human review conclusion",
            "",
            "Best halo:",
            "",
            "Best beam preset:",
            "",
            "Best target view size:",
            "",
            "Best context margin:",
            "",
            "Observed failure modes:",
            "",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _human_judgment() -> dict[str, Any]:
    return {
        "target_covered": None,
        "all_relevant_regions_covered": None,
        "context_sufficient": None,
        "redundancy_reasonable": None,
        "zoom_reasonable": None,
        "overall": None,
        "notes": "",
    }


def main() -> int:
    args = _parse_args()
    base_config_path = Path(args.base_config).expanduser().resolve()
    experiment = load_experiment_config(args.experiment_config)
    manifest = load_sample_manifest(args.manifest)
    base_config = load_locator_config(base_config_path)
    experiment_base = deep_merge(base_config, experiment.get("base_overrides", {}))
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_presets = set(args.preset or experiment["presets"].keys())
    unknown_presets = selected_presets.difference(experiment["presets"])
    if unknown_presets:
        raise LocatorError(f"unknown presets: {sorted(unknown_presets)}")
    selected_sample_ids = set(args.sample_id or [])
    samples = [
        sample
        for sample in manifest["samples"]
        if not selected_sample_ids or sample["id"] in selected_sample_ids
    ]
    if selected_sample_ids.difference(sample["id"] for sample in samples):
        raise LocatorError("one or more --sample-id values are absent from the manifest")
    max_samples = int(experiment.get("max_samples", 5))
    if not args.sample_id:
        samples = samples[:max_samples]
    if not samples:
        raise LocatorError("no samples selected")

    experiment_manifest = {
        "schema_version": "uhr-locator-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config": {"path": str(base_config_path), "sha256": _sha256(base_config_path)},
        "experiment_config": {
            "path": experiment["_source_path"],
            "sha256": experiment["_source_sha256"],
        },
        "sample_manifest": {
            "path": manifest["_source_path"],
            "sha256": manifest["_source_sha256"],
            "split": manifest.get("split"),
            "usage": manifest.get("usage"),
        },
        "presets": sorted(selected_presets),
        "samples": [dict(sample) for sample in samples],
        "provider_cli_overrides": {
            "detector_provider": args.detector_provider,
            "retriever_provider": args.retriever_provider,
            "enable_detector": args.enable_detector,
            "disable_detector": args.disable_detector,
            "enable_retriever": args.enable_retriever,
            "disable_retriever": args.disable_retriever,
        },
        "command": [sys.executable, *sys.argv],
    }
    _write_json(output_root / "experiment_manifest.json", experiment_manifest)

    rows: list[dict[str, Any]] = []
    for preset_name, preset in experiment["presets"].items():
        if preset_name not in selected_presets:
            continue
        config = deep_merge(experiment_base, preset.get("overrides", {}))
        _apply_provider_overrides(config, args)
        locator_name = str(config.get("locator", {}).get("provider", "hierarchical"))
        try:
            locator = create_locator(locator_name, config)
        except Exception as exc:
            _write_json(
                output_root / preset_name / "failure.json",
                {
                    "status": "failed",
                    "provider_stage": "locator_initialization",
                    "provider_error": f"{type(exc).__name__}: {exc}",
                    "preset": preset_name,
                    "config": config,
                    "command": [sys.executable, *sys.argv],
                },
            )
            raise
        try:
            for sample in samples:
                sample_id = str(sample["id"])
                artifact_dir = output_root / preset_name / sample_id
                result_path = artifact_dir / "result.json"
                if (args.resume or args.skip_existing) and result_path.is_file():
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "preset": preset_name,
                            "sample_id": sample_id,
                            "artifact_dir": str(artifact_dir),
                            **existing["diagnostic_metrics"],
                        }
                    )
                    continue
                image_path = _resolve_environment_path(
                    sample["image"],
                    label=f"sample {sample_id} image",
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                try:
                    result = locator.locate(image_path, str(sample["question"]))
                except Exception as exc:
                    _write_json(
                        artifact_dir / "failure.json",
                        {
                            "status": "failed",
                            "provider_stage": "locator",
                            "provider_error": f"{type(exc).__name__}: {exc}",
                            "preset": preset_name,
                            "sample_id": sample_id,
                            "config": config,
                            "command": [sys.executable, *sys.argv],
                        },
                    )
                    raise
                payload = result.to_dict()
                _write_json(artifact_dir / "search_trace.json", payload["search_trace"])
                exports = render_sample_artifacts(image_path, payload, sample, artifact_dir)
                with Image.open(image_path) as image:
                    image_size = image.size
                metrics = collect_metrics(payload, sample, image_size, exports["detector"])
                saved = {
                    **payload,
                    "preset": preset_name,
                    "preset_description": preset.get("description", ""),
                    "sample": dict(sample),
                    "image_path": str(image_path),
                    "effective_config": config,
                    "exports": {key: value for key, value in exports.items() if key != "detector"},
                    "diagnostic_metrics": metrics,
                    "human_judgment": _human_judgment(),
                }
                _write_json(result_path, saved)
                rows.append(
                    {
                        "preset": preset_name,
                        "sample_id": sample_id,
                        "artifact_dir": str(artifact_dir),
                        **metrics,
                    }
                )
        finally:
            locator.close()
    _write_summary(output_root, rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_root),
                "preset_count": len(selected_presets),
                "sample_count": len(samples),
                "result_count": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
