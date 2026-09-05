#!/usr/bin/env python3
"""Run the query-aware UHR locator and emit a complete JSON trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.locators.config import load_locator_config  # noqa: E402
from sat_rs_vlm.integrations.locators.registry import create_locator  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output")
    parser.add_argument("--detector-provider")
    parser.add_argument("--retriever-provider")
    detector = parser.add_mutually_exclusive_group()
    detector.add_argument("--enable-detector", action="store_true")
    detector.add_argument("--disable-detector", action="store_true")
    retriever = parser.add_mutually_exclusive_group()
    retriever.add_argument("--enable-retriever", action="store_true")
    retriever.add_argument("--disable-retriever", action="store_true")
    parser.add_argument("--export-crops")
    parser.add_argument("--export-debug-overlay")
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    detector = config.setdefault("detector", {})
    retriever = config.setdefault("retriever", {})
    scorers = config.setdefault("scorers", {})
    detector_scorer = scorers.setdefault("detector", {})
    retriever_scorer = scorers.setdefault("retrieval", {})
    if args.detector_provider:
        detector["provider"] = args.detector_provider
        detector["enabled"] = True
        detector_scorer["enabled"] = True
    if args.retriever_provider:
        retriever["provider"] = args.retriever_provider
        retriever["enabled"] = True
        retriever_scorer["enabled"] = True
    if args.enable_detector:
        detector["enabled"] = True
        detector_scorer["enabled"] = True
    if args.disable_detector:
        detector["enabled"] = False
        detector_scorer["enabled"] = False
    if args.enable_retriever:
        retriever["enabled"] = True
        retriever_scorer["enabled"] = True
    if args.disable_retriever:
        retriever["enabled"] = False
        retriever_scorer["enabled"] = False


def _export_crops(image_path: Path, payload: dict[str, Any], output_dir: Path) -> list[str]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    with Image.open(image_path) as image:
        for index, box in enumerate(payload["regions_xyxy"]):
            crop_path = output_dir / f"region_{index:03d}.png"
            image.crop(tuple(round(float(value)) for value in box)).save(crop_path)
            exported.append(str(crop_path.resolve()))
    return exported


def _export_overlay(image_path: Path, payload: dict[str, Any], output_path: Path) -> str:
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in payload["search_trace"]:
        core = tuple(float(value) for value in item["core_xyxy"])
        view = tuple(float(value) for value in item["view_xyxy"])
        selected = bool(item["selected"])
        draw.rectangle(view, outline=(230, 180, 30), width=1)
        draw.rectangle(core, outline=(220, 30, 30) if selected else (100, 100, 100), width=2)
        if selected:
            label = f"d{item['depth']} {item['fused_score']:.3f}"
            draw.text((core[0] + 3, core[1] + 3), label, fill=(255, 255, 255))
    image.save(output_path)
    return str(output_path.resolve())


def main() -> int:
    args = _parse_args()
    config = load_locator_config(args.config)
    _apply_overrides(config, args)
    locator_name = str(config.get("locator", {}).get("provider", "hierarchical"))
    locator = create_locator(locator_name, config)
    try:
        result = locator.locate(Path(args.image), args.question)
        payload = result.to_dict()
        exports: dict[str, Any] = {}
        if args.export_crops:
            exports["crops"] = _export_crops(
                Path(args.image), payload, Path(args.export_crops)
            )
        if args.export_debug_overlay:
            exports["debug_overlay"] = _export_overlay(
                Path(args.image), payload, Path(args.export_debug_overlay)
            )
        if exports:
            payload["exports"] = exports
        encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
    finally:
        locator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
