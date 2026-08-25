#!/usr/bin/env python3
"""Isolated LAE-DINO/MMDetection JSONL proposal worker."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.detectors.protocol import (  # noqa: E402
    ProposalError,
    canonicalize_proposals,
)


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((left[2] - left[0]) * (left[3] - left[1])) + (
        (right[2] - right[0]) * (right[3] - right[1])
    ) - intersection
    return intersection / union if union > 0.0 else 0.0


def _nms(boxes: list[list[float]], scores: list[float], threshold: float) -> tuple[list[list[float]], list[float]]:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [index for index in order if _iou(boxes[current], boxes[index]) <= threshold]
    return [boxes[index] for index in keep], [scores[index] for index in keep]


def _insert_source_root(source_root: Path) -> None:
    for candidate in (source_root, source_root / "mmdetection_lae"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _extract_predictions(result: Any) -> tuple[list[Any], list[Any]]:
    """Support MMDetection 2 tuple results and MMDetection 3 DataSamples."""

    if hasattr(result, "pred_instances"):
        instances = result.pred_instances
        boxes = getattr(instances, "bboxes", [])
        scores = getattr(instances, "scores", [])
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().tolist()
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().tolist()
        return list(boxes), list(scores)
    if isinstance(result, (list, tuple)):
        # MMDetection 2.x returns ``(bbox_results, segmentation_results)``;
        # only the first component is detector box evidence.
        class_results = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        all_boxes: list[Any] = []
        all_scores: list[Any] = []
        for class_result in class_results:
            if hasattr(class_result, "detach"):
                class_result = class_result.detach().cpu().tolist()
            for row in class_result or []:
                try:
                    values = list(row)
                    if len(values) >= 5:
                        all_boxes.append(values[:4])
                        all_scores.append(values[4])
                except TypeError:
                    continue
        return all_boxes, all_scores
    raise ProposalError(f"unsupported LAE-DINO inference result type: {type(result)!r}")


def _load_detector(args: argparse.Namespace) -> Any:
    source_root = Path(args.source_root).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path, label in ((source_root, "source_root"), (config, "config"), (checkpoint, "checkpoint")):
        if not path.exists():
            raise RuntimeError(f"LAE-DINO {label} does not exist: {path}")
    if args.bert_root:
        bert_root = Path(args.bert_root).expanduser().resolve()
        if not bert_root.is_dir():
            raise RuntimeError(f"LAE-DINO BERT root does not exist: {bert_root}")
        os.environ["LAE_DINO_BERT_ROOT"] = str(bert_root)
    _insert_source_root(source_root)
    try:
        from mmdet.apis import init_detector
    except ImportError as exc:
        raise RuntimeError("LAE-DINO sidecar requires a compatible mmdet installation") from exc
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = init_detector(str(config), str(checkpoint), device=args.device)
    return model


def _predict(model: Any, request: dict[str, Any], args: argparse.Namespace, inference_detector: Any) -> dict[str, Any]:
    from PIL import Image

    image_path = Path(str(request["image"])).expanduser().resolve()
    image = Image.open(str(image_path)).convert("RGB")
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = inference_detector(model, str(image_path))
    boxes, scores = _extract_predictions(result)
    filtered = [
        (box, score)
        for box, score in zip(boxes, scores, strict=True)
        if float(score) >= args.score_threshold
    ]
    boxes = [item[0] for item in filtered]
    scores = [item[1] for item in filtered]
    boxes, scores, stats = canonicalize_proposals(
        boxes,
        scores,
        image_width=image.width,
        image_height=image.height,
        coordinate_mode="pixel",
        top_k=args.top_k,
    )
    if args.nms_threshold is not None:
        boxes, scores = _nms(boxes, scores, args.nms_threshold)
        boxes, scores = boxes[: args.top_k], scores[: args.top_k]
    return {
        "id": request["id"],
        "status": "ok",
        "bbox_list": boxes,
        "bbox_scores": scores,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "metadata": {
            "schema_version": "lae-dino-sidecar-v1",
            "target_phrase": str(request.get("target_phrase", "")).strip().lower(),
            "target_phrase_used_for_detector": False,
            "closed_set_detector": True,
            "score_threshold": args.score_threshold,
            "top_k": args.top_k,
            "nms_threshold": args.nms_threshold,
            "coordinate_mode": "absolute_pixel_xyxy",
            "image_width": image.width,
            "image_height": image.height,
            "validation": stats,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bert-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--nms-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = _load_detector(args)
        # Import only after model initialization; custom LAE-DINO mmdet APIs can
        # expose inference_detector from different namespaces.
        try:
            from mmdet.apis import inference_detector
        except ImportError as exc:
            raise RuntimeError("compatible mmdet inference_detector is unavailable") from exc
    except Exception as exc:
        print(json.dumps({"status": "failed", "failure_stage": "model_init", "error": str(exc)}), flush=True)
        return 2
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            for key in ("id", "image", "target_phrase"):
                if key not in request:
                    raise ValueError(f"missing request field: {key}")
            response = _predict(model, request, args, inference_detector)
        except json.JSONDecodeError as exc:
            response = {"status": "failed", "failure_stage": "protocol_guard", "error": str(exc)}
        except Exception as exc:
            response = {
                "id": request.get("id") if isinstance(locals().get("request"), dict) else None,
                "status": "failed",
                "failure_stage": "proposal_generation",
                "error": str(exc),
            }
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
