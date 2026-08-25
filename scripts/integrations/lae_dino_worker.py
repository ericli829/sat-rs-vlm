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


def _nms(
    boxes: list[list[float]], scores: list[float], threshold: float
) -> tuple[list[list[float]], list[float]]:
    if threshold < 0.0 or threshold > 1.0:
        raise ProposalError("nms_threshold must be between 0 and 1")
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


def _validate_local_bert_root(bert_root: Path) -> None:
    """Fail before MMDetection/HF construction when the offline BERT is incomplete."""

    required = [bert_root / "config.json"]
    if not any((bert_root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        required.append(bert_root / "model.safetensors")
    if not any((bert_root / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        required.append(bert_root / "tokenizer.json")
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "LAE-DINO local BERT root is incomplete; missing "
            + ", ".join(missing)
            + f": {bert_root}"
        )


def _patch_lae_config_for_local_runtime(config: Any, bert_root: Path) -> Any:
    """Patch only the in-memory config used by the detector sidecar.

    LAE's legacy BERT builder calls ``from_pretrained`` on
    ``model.language_model.name``.  Relative config paths therefore depend on
    cwd and can trigger an unwanted Hub lookup.  An explicit local directory
    is required and is written into the MMEngine config object before model
    construction; no checkpoint/config file on disk is modified.
    """

    model = config.get("model") if hasattr(config, "get") else getattr(config, "model", None)
    if model is None:
        raise RuntimeError("LAE-DINO config has no model section")
    language_model = (
        model.get("language_model")
        if hasattr(model, "get")
        else getattr(model, "language_model", None)
    )
    if language_model is None:
        raise RuntimeError("LAE-DINO config has no model.language_model section")
    if hasattr(language_model, "__setitem__"):
        language_model["name"] = str(bert_root)
    else:
        language_model.name = str(bert_root)

    # With an explicit detector checkpoint, do not attempt config-level
    # bootstrap URLs before the checkpoint is loaded.
    if hasattr(config, "__setitem__"):
        if config.get("load_from"):
            config["load_from"] = None
    elif getattr(config, "load_from", None):
        config.load_from = None
    backbone = (
        model.get("backbone")
        if hasattr(model, "get")
        else getattr(model, "backbone", None)
    )
    if backbone is not None:
        init_cfg = (
            backbone.get("init_cfg")
            if hasattr(backbone, "get")
            else getattr(backbone, "init_cfg", None)
        )
        if init_cfg:
            if hasattr(backbone, "__setitem__"):
                backbone["init_cfg"] = None
            else:
                backbone.init_cfg = None
    return config


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
    for path, label in (
        (source_root, "source_root"),
        (config, "config"),
        (checkpoint, "checkpoint"),
    ):
        if not path.exists():
            raise RuntimeError(f"LAE-DINO {label} does not exist: {path}")
    if not args.bert_root:
        raise RuntimeError("LAE-DINO requires --bert-root for offline BERT loading")
    bert_root = Path(args.bert_root).expanduser().resolve()
    if not bert_root.is_dir():
        raise RuntimeError(f"LAE-DINO BERT root does not exist: {bert_root}")
    _validate_local_bert_root(bert_root)
    os.environ["LAE_DINO_BERT_ROOT"] = str(bert_root)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    _insert_source_root(source_root)
    try:
        from mmdet.apis import init_detector
        from mmengine.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "LAE-DINO sidecar requires compatible mmengine/mmdet installations"
        ) from exc
    cfg = Config.fromfile(str(config))
    cfg = _patch_lae_config_for_local_runtime(cfg, bert_root)
    # Keep stdout reserved for JSONL.  Leave stderr untouched so the parent
    # sidecar can preserve library diagnostics in its stderr log.
    with contextlib.redirect_stdout(io.StringIO()):
        model = init_detector(cfg, str(checkpoint), device=args.device)
    return model


def _predict(
    model: Any,
    request: dict[str, Any],
    args: argparse.Namespace,
    inference_detector: Any,
) -> dict[str, Any]:
    from PIL import Image

    image_path = Path(str(request["image"])).expanduser().resolve()
    image = Image.open(str(image_path)).convert("RGB")
    started = time.perf_counter()
    # Keep detector/library diagnostics on stderr; the sidecar captures it.
    with contextlib.redirect_stdout(io.StringIO()):
        query = str(request.get("target_phrase", "")).strip().lower()
        if not query:
            raise ValueError("target_phrase must not be empty")
        # Pinned LAE-DINO exposes text_prompt/custom_entities on its inference
        # API.  This is target-conditioned inference, even for fine-tuned
        # checkpoints whose training regime is dataset-specific.
        result = inference_detector(
            model,
            str(image_path),
            text_prompt=query,
            custom_entities=True,
        )
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
        # Apply final top-k after optional NMS, never before suppression.
        top_k=None,
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
            "target_phrase": query,
            "target_phrase_used_for_detector": True,
            "custom_entities": True,
            "checkpoint_training_regime": getattr(
                args, "checkpoint_training_regime", "unspecified"
            ),
            "source_revision": getattr(args, "source_revision", "unspecified"),
            "inference_query_mode": getattr(
                args, "inference_query_mode", "target_conditioned_text_prompt"
            ),
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
    parser.add_argument("--checkpoint-training-regime", default="unspecified")
    parser.add_argument("--source-revision", default="unspecified")
    parser.add_argument(
        "--inference-query-mode", default="target_conditioned_text_prompt"
    )
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
        print(
            json.dumps(
                {"status": "failed", "failure_stage": "model_init", "error": str(exc)}
            ),
            flush=True,
        )
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
