"""Run DINO proposals, fixed-category GeoRSCLIP screening, then attribute scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.detectors.dino_georsclip import (  # noqa: E402
    DinoGeoRSCLIPTwoStage,
    resolve_target,
)
from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider  # noqa: E402
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult  # noqa: E402
from sat_rs_vlm.integrations.retrievers.registry import create_retriever_provider  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/integrations/dino_georsclip_two_stage.example.yaml"
DEFAULT_PROMPTS = PROJECT_ROOT / "configs/prompts/georsclip_major_categories.yaml"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Source image to detect")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target", help="Object phrase or counting question")
    target.add_argument("--question", help="Counting question; category is inferred from it")
    parser.add_argument("--modifier", help="Attribute to select, for example red or large")
    parser.add_argument(
        "--contrast-modifier",
        help="Alternative attribute; defaults to non-<modifier>, for example non-red",
    )
    parser.add_argument("--unrestricted-query", help="Override the unmodified category prompt")
    parser.add_argument("--qualified-query", help="Override the attribute target prompt")
    parser.add_argument("--contrast-query", help="Override the attribute contrast prompt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt-catalog", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--proposal-json",
        type=Path,
        help="Reuse DINO boxes from a saved result JSON instead of starting the detector",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coarse-margin-threshold", type=float)
    parser.add_argument("--qualified-margin-threshold", type=float)
    return parser.parse_args()


class SavedDinoProposalProvider:
    """Replay existing DINO boxes while keeping the same downstream pipeline."""

    provider_name = "saved_dino_json"

    def __init__(self, result_path: Path, image_path: Path) -> None:
        self.result_path = result_path.expanduser().resolve()
        if not self.result_path.is_file():
            raise FileNotFoundError(self.result_path)
        self.data = json.loads(self.result_path.read_text(encoding="utf-8"))
        saved_image = self.data.get("source_image", self.data.get("image"))
        if saved_image:
            saved_path = Path(str(saved_image)).expanduser().resolve()
            requested_path = image_path.expanduser().resolve()
            if str(saved_path).casefold() != str(requested_path).casefold():
                raise ValueError(
                    "proposal JSON belongs to a different image: "
                    f"{saved_path} != {requested_path}"
                )
        raw_rows = self.data.get("candidates")
        if not isinstance(raw_rows, list):
            raise ValueError("proposal JSON must contain a candidates list")
        self.boxes: list[list[float]] = []
        self.scores: list[float] = []
        for index, row in enumerate(raw_rows):
            if not isinstance(row, dict) or "bbox_xyxy" not in row:
                raise ValueError(f"proposal JSON candidate {index} has no bbox_xyxy")
            score = row.get("dino_score", row.get("det_score", row.get("score")))
            if score is None:
                raise ValueError(f"proposal JSON candidate {index} has no detector score")
            self.boxes.append([float(value) for value in row["bbox_xyxy"]])
            self.scores.append(float(score))
        self.threshold = self.data.get(
            "selected_box_threshold", self.data.get("box_threshold")
        )

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        return ProposalResult(
            boxes_xyxy=[list(box) for box in self.boxes],
            scores=list(self.scores),
            latency_ms=0.0,
            provider=self.provider_name,
            model_id=self.result_path.name,
            metadata={
                "replayed_from": str(self.result_path),
                "source_image": str(Path(image_path).resolve()),
                "original_detection_json": self.data.get("detection_json"),
                "saved_detector_prompt": self.data.get("target_category"),
                "reused_box_threshold": self.threshold,
                "current_target_phrase_not_rerun": target_phrase,
                "candidate_pool_complete": True,
            },
        )

    def close(self) -> None:
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return payload


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return safe or "target"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default()


def _square_crop(source: Image.Image, bbox: list[float], size: int) -> Image.Image:
    x0, y0, x1, y1 = bbox
    box = (
        max(0, math.floor(x0)),
        max(0, math.floor(y0)),
        min(source.width, math.ceil(x1)),
        min(source.height, math.ceil(y1)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        crop = Image.new("RGB", (size, size), (127, 127, 127))
    else:
        crop = source.crop(box).convert("RGB")
    side = max(crop.size)
    square = Image.new("RGB", (side, side), (127, 127, 127))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    crop.close()
    result = square.resize((size, size), Image.Resampling.LANCZOS)
    square.close()
    return result


def _make_sheet(
    source: Image.Image,
    rows: list[dict[str, Any]],
    *,
    title: str,
    score_key: str,
    output: Path,
) -> None:
    columns, cell_width, image_size, text_height = 8, 150, 112, 40
    padding, title_height, gap = 12, 82, 8
    row_count = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new(
        "RGB",
        (
            2 * padding + columns * cell_width,
            title_height + padding + row_count * (image_size + text_height + gap) + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 13), title, font=_font(22), fill=(20, 20, 20))
    draw.text(
        (padding, 49),
        f"结果 {len(rows)} 个；按分数排序；相似度与分差不是概率。",
        font=_font(15),
        fill=(65, 65, 65),
    )
    for index, row in enumerate(rows):
        column, grid_row = index % columns, index // columns
        x = padding + column * cell_width
        y = title_height + padding + grid_row * (image_size + text_height + gap)
        crop = _square_crop(source, row["bbox_xyxy"], image_size)
        canvas.paste(crop, (x + (cell_width - image_size) // 2, y))
        crop.close()
        is_qualified_score = score_key == "combined_score"
        rank = row.get("qualified_rank") if is_qualified_score else row.get("coarse_rank")
        score = row.get(score_key)
        draw.text(
            (x + 2, y + image_size + 1),
            f"#{int(rank or 0):03d} ID {int(row['id']):03d} D{float(row['dino_score']):.2f}",
            font=_font(12),
            fill=(25, 25, 25),
        )
        draw.text(
            (x + 2, y + image_size + 19),
            f"综合分 {float(score):+.4f}" if is_qualified_score else f"大类分差 {float(score):+.4f}",
            font=_font(11),
            fill=(65, 65, 65),
        )
    canvas.save(output)
    canvas.close()


def _make_overlay(
    source: Image.Image,
    all_rows: list[dict[str, Any]],
    selected_ids: set[int],
    *,
    title: str,
    output: Path,
) -> None:
    header_height = 82
    canvas = Image.new("RGB", (source.width, source.height + header_height), "white")
    canvas.paste(source, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, font=_font(22), fill=(20, 20, 20))
    draw.text(
        (14, 44),
        f"红框：保留 {len(selected_ids)} 个；浅白框：其余 DINO 候选，共 {len(all_rows)} 个。",
        font=_font(15),
        fill=(55, 55, 55),
    )
    ordered = sorted(
        all_rows,
        key=lambda row: max(0.0, row["bbox_xyxy"][2] - row["bbox_xyxy"][0])
        * max(0.0, row["bbox_xyxy"][3] - row["bbox_xyxy"][1]),
        reverse=True,
    )
    for row in ordered:
        x0, y0, x1, y1 = row["bbox_xyxy"]
        x0 = max(0, min(source.width - 1, math.floor(x0)))
        y0 = max(0, min(source.height - 1, math.floor(y0))) + header_height
        x1 = max(0, min(source.width - 1, math.ceil(x1)))
        y1 = max(0, min(source.height - 1, math.ceil(y1))) + header_height
        if x1 <= x0 or y1 <= y0:
            continue
        color = (220, 25, 32) if int(row["id"]) in selected_ids else (246, 246, 239)
        draw.rectangle((x0, y0, x1, y1), outline=(25, 25, 25), width=6)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    canvas.save(output)
    canvas.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "bbox_xyxy",
        "dino_score",
        "coarse_positive_score",
        "coarse_negative_score",
        "coarse_margin",
        "coarse_selected",
        "coarse_rank",
        "qualified_score",
        "unrestricted_score",
        "contrast_score",
        "delta_qualified_minus_unrestricted",
        "delta_qualified_minus_contrast",
        "mixed_attribute_score",
        "combined_score",
        "combined_selected",
        "combined_rank",
        "qualified_selected",
        "qualified_rank",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["bbox_xyxy"] = json.dumps(row["bbox_xyxy"], ensure_ascii=False)
            writer.writerow(encoded)


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(result["candidates"])
    coarse_ranked = sorted(candidates, key=lambda row: int(row["coarse_rank"] or 0))
    unqualified = list(result["unqualified_results"])
    qualified = list(result["qualified_results"])
    (output_dir / "two_stage_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "all_dino_candidates.csv", coarse_ranked)
    _write_csv(output_dir / "unqualified_results.csv", unqualified)
    _write_csv(output_dir / "qualified_results.csv", qualified)
    with Image.open(result["image"]) as source_file:
        source = source_file.convert("RGB")
    _make_sheet(
        source,
        coarse_ranked,
        title="DINO 候选框：GeoRSCLIP 大类分差（高 → 低）",
        score_key="coarse_margin",
        output=output_dir / "all_dino_candidates_ranked.png",
    )
    _make_sheet(
        source,
        unqualified,
        title="无限定词结果：DINO + GeoRSCLIP 大类粗筛",
        score_key="coarse_margin",
        output=output_dir / "unqualified_results.png",
    )
    _make_sheet(
        source,
        qualified,
        title="有限定词结果：大类分差 + 限定词分差综合排序（高 → 低）",
        score_key="combined_score",
        output=output_dir / "qualified_results.png",
    )
    _make_overlay(
        source,
        candidates,
        {int(row["id"]) for row in unqualified},
        title="无限定词：大类正负提示词粗筛",
        output=output_dir / "unqualified_source_overlay.png",
    )
    _make_overlay(
        source,
        candidates,
        {int(row["id"]) for row in qualified},
        title="有限定词：大类分差 > 0 且 80/20 限定词分差 > 0",
        output=output_dir / "qualified_source_overlay.png",
    )
    source.close()


def main() -> None:
    args = _parse_args()
    config = _load_yaml(args.config)
    prompt_catalog = _load_yaml(args.prompt_catalog)
    raw_target = args.question or args.target or ""
    target_category, group_name, inferred_modifier = resolve_target(
        raw_target,
        prompt_catalog.get("groups", {}),
        modifier_override=args.modifier,
    )
    modifier = args.modifier or inferred_modifier
    if not modifier:
        raise ValueError(
            "an attribute modifier is required; pass --modifier red or include it "
            "before the target in --target/--question"
        )
    detector_section = config.get("detector", {})
    retriever_section = config.get("retriever", {})
    if not isinstance(detector_section, dict) or not isinstance(retriever_section, dict):
        raise ValueError("config must define detector and retriever mappings")
    detector = (
        SavedDinoProposalProvider(args.proposal_json, args.image)
        if args.proposal_json
        else create_proposal_provider(
            str(detector_section.get("provider", "")),
            detector_section.get("config", {}),
        )
    )
    retriever = create_retriever_provider(
        str(retriever_section.get("provider", "georsclip")),
        retriever_section.get("config", {}),
    )
    try:
        coarse_config = config.get("coarse_filter", {})
        attribute_config = config.get("attribute_filter", {})
        coarse_threshold = (
            args.coarse_margin_threshold
            if args.coarse_margin_threshold is not None
            else float(coarse_config.get("margin_threshold", 0.0))
        )
        qualified_threshold = (
            args.qualified_margin_threshold
            if args.qualified_margin_threshold is not None
            else float(attribute_config.get("target_margin_threshold", 0.0))
        )
        strategy = DinoGeoRSCLIPTwoStage(
            detector,
            retriever,
            prompt_catalog,
            coarse_margin_threshold=coarse_threshold,
            qualified_margin_threshold=qualified_threshold,
            unrestricted_weight=float(attribute_config.get("unrestricted_weight", 0.80)),
            contrast_weight=float(attribute_config.get("contrast_weight", 0.20)),
            coarse_score_weight=float(attribute_config.get("coarse_score_weight", 0.50)),
            attribute_score_weight=float(attribute_config.get("attribute_score_weight", 0.50)),
        )
        result = strategy.run(
            args.image,
            target_category=target_category,
            group_name=group_name,
            modifier=modifier,
            contrast_modifier=args.contrast_modifier,
            unrestricted_query=args.unrestricted_query,
            qualified_query=args.qualified_query,
            contrast_query=args.contrast_query,
        )
    finally:
        detector.close()
        retriever.close()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "reports"
            / "evaluation"
            / "dino_georsclip_two_stage"
            / f"{args.image.stem}_{_safe_name(target_category)}_{_safe_name(modifier)}"
        )
    _write_outputs(result, output_dir)
    print(
        json.dumps(
            {
                "image": result["image"],
                "target_category": target_category,
                "prompt_group": group_name,
                "modifier": modifier,
                "dino_candidates": len(result["candidates"]),
                "unqualified_detections": len(result["unqualified_results"]),
                "qualified_detections": len(result["qualified_results"]),
                "output_dir": str(output_dir.resolve()),
                "attribute_formula": result["attribute_filter"]["formula"],
                "combined_formula": result["combined_ranking"]["formula"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
