#!/usr/bin/env python3
"""Isolated LAE-DINO/MMDetection JSONL proposal worker."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import io
import json
import math
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
    union = (
        ((left[2] - left[0]) * (left[3] - left[1]))
        + ((right[2] - right[0]) * (right[3] - right[1]))
        - intersection
    )
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
    backbone = model.get("backbone") if hasattr(model, "get") else getattr(model, "backbone", None)
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
        # Detector-only proposal inference has no validation dataset.  The
        # legacy LAE-DINO API otherwise builds ``test_dataloader`` solely to
        # discover a palette, forcing unrelated DIOR/DOTA annotations.
        # The vendored API's ``palette='none'`` branch builds the full test
        # dataset; ``random`` selects a visualization-only palette directly.
        model = init_detector(cfg, str(checkpoint), device=args.device, palette="random")
    return model


def _release_request_resources() -> None:
    gc.collect()
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(getattr(cuda, "is_available", lambda: False)()):
            cuda.empty_cache()
    except Exception:
        pass
    try:
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if callable(trim):
            trim(0)
    except Exception:
        pass


def _predict(
    model: Any,
    request: dict[str, Any],
    args: argparse.Namespace,
    inference_detector: Any,
) -> dict[str, Any]:
    from PIL import Image

    image_path = Path(str(request["image"])).expanduser().resolve()
    with Image.open(str(image_path)) as source:
        image_width, image_height = source.size
    started = time.perf_counter()
    args = _request_options(args, request)
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
    return _prediction_response(result, request, args, image_width, image_height, started)


def _request_options(args: argparse.Namespace, request: dict[str, Any]) -> argparse.Namespace:
    """Postprocessing is request-local so LOCATE and COUNT may share one model."""
    options = request.get("options", {})
    allowed = {"score_threshold", "top_k", "nms_threshold"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("invalid detector request options")
    current = argparse.Namespace(**vars(args))
    for key, value in options.items():
        setattr(current, key, value)
    current.score_threshold = float(current.score_threshold)
    if not math.isfinite(current.score_threshold) or not 0 <= current.score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")
    if isinstance(current.top_k, bool) or int(current.top_k) != current.top_k or current.top_k < 1:
        raise ValueError("top_k must be a positive integer")
    current.top_k = int(current.top_k)
    if current.nms_threshold is not None:
        current.nms_threshold = float(current.nms_threshold)
        if not math.isfinite(current.nms_threshold) or not 0 <= current.nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
    return current


def _prediction_response(
    result: Any,
    request: dict[str, Any],
    args: argparse.Namespace,
    image_width: int,
    image_height: int,
    started: float,
) -> dict[str, Any]:
    query = str(request["target_phrase"]).strip().lower()
    boxes, scores = _extract_predictions(result)
    if len(boxes) != len(scores):
        raise ProposalError(f"LAE-DINO boxes/scores length mismatch: {len(boxes)} != {len(scores)}")
    filtered = [
        (box, score)
        # The isolated official LAE runtime uses Python 3.9.  Length equality
        # is checked above instead of relying on Python 3.10's zip(strict=...).
        for box, score in zip(boxes, scores)  # noqa: B905 - Python 3.9 sidecar
        if float(score) >= args.score_threshold
    ]
    boxes = [item[0] for item in filtered]
    scores = [item[1] for item in filtered]
    boxes, scores, stats = canonicalize_proposals(
        boxes,
        scores,
        image_width=image_width,
        image_height=image_height,
        coordinate_mode="pixel",
        # Apply final top-k after optional NMS, never before suppression.
        top_k=None,
    )
    if args.nms_threshold is not None:
        boxes, scores = _nms(boxes, scores, args.nms_threshold)
    candidate_count_before_top_k = len(boxes)
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
            "candidate_count_before_top_k": candidate_count_before_top_k,
            "candidate_truncated": candidate_count_before_top_k > len(boxes),
            "candidate_pool_complete": candidate_count_before_top_k == len(boxes)
            and stats["invalid_count"] == 0,
            "nms_threshold": args.nms_threshold,
            "coordinate_mode": "absolute_pixel_xyxy",
            "image_width": image_width,
            "image_height": image_height,
            "validation": stats,
        },
    }


def _build_batch_inference(model: Any) -> Any:
    """Build the original pipeline once; call test_step ONCE for a whole batch.

    The vendored inference_detector(list) loops over singleton test_step calls.
    Keep its pipeline, BGR/RGB policy and rescaling metadata, bypass only that loop.
    """
    import torch
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg

    pipeline = Compose(get_test_pipeline_cfg(model.cfg.copy()))

    def infer(requests: list[dict[str, Any]]) -> list[Any]:
        inputs, samples = [], []
        for index, request in enumerate(requests):
            data = pipeline(
                dict(
                    img_path=str(Path(request["image"]).expanduser().resolve()),
                    img_id=index,
                    text=str(request["target_phrase"]).strip().lower(),
                    custom_entities=True,
                )
            )
            inputs.append(data["inputs"])
            samples.append(data["data_samples"])
        with torch.no_grad():
            return model.test_step(dict(inputs=inputs, data_samples=samples))

    return infer


def _is_cuda_oom(exc: Exception) -> bool:
    # Also supports torch 1.x, which may not expose torch.cuda.OutOfMemoryError.
    return isinstance(exc, RuntimeError) and (
        "cuda out of memory" in str(exc).lower()
        or "cuda error: out of memory" in str(exc).lower()
        or type(exc).__name__ == "OutOfMemoryError"
    )


def _predict_batch(
    model: Any,
    request: dict[str, Any],
    args: argparse.Namespace,
    batch_inference: Any,
) -> dict[str, Any]:
    from PIL import Image

    started = time.perf_counter()
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("predict_batch requires nonempty items")
    dimensions, options = [], []
    ids = []
    for item in items:
        required = ("id", "image", "target_phrase")
        if not isinstance(item, dict) or any(k not in item for k in required):
            raise ValueError("batch item requires id/image/target_phrase")
        if not isinstance(item["id"], str) or not str(item["target_phrase"]).strip():
            raise ValueError("batch item requires string id and nonempty target_phrase")
        ids.append(item["id"])
        options.append(_request_options(args, item))
        with Image.open(str(item["image"])) as image:
            dimensions.append(image.size)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate batch item ids")
    limit = min(args.max_batch_size, getattr(args, "_safe_batch_size", args.max_batch_size))
    chunked = int(getattr(model, "test_cfg", {}).get("chunked_size", -1)) > 0
    if chunked:
        limit = 1
    offset, retries = 0, 0
    responses = []
    while offset < len(items):
        chunk = items[offset : offset + limit]
        oom = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                predictions = batch_inference(chunk)
            if not isinstance(predictions, (list, tuple)) or len(predictions) != len(chunk):
                raise ProposalError("batch forward result count mismatch")
            current = []
            for index, prediction in enumerate(predictions):
                item_index = offset + index
                width, height = dimensions[item_index]
                response = _prediction_response(
                    prediction, items[item_index], options[item_index], width, height, started
                )
                response["metadata"]["effective_batch_size"] = len(chunk)
                current.append(response)
        except Exception as exc:
            if not _is_cuda_oom(exc) or args.disable_batch_oom_retry or len(chunk) == 1:
                raise
            oom = True
        # Leave the exception scope (and its traceback) before freeing GPU buffers.
        if oom:
            predictions = None
            current = None
            prediction = None
            _release_request_resources()
            limit = max(1, len(chunk) // 2)
            args._safe_batch_size = limit
            retries += 1
            continue
        responses.extend(current)
        predictions = current = prediction = None
        offset += len(chunk)
    elapsed = (time.perf_counter() - started) * 1000.0
    for response in responses:
        response["latency_ms"] = elapsed / len(items)
        response["metadata"].update(
            batch_size_requested=len(items),
            batch_worker_ms=elapsed,
            batch_oom_retries=retries,
            batch_limit=limit,
            chunked_text_serial=chunked,
            worker_pid=os.getpid(),
        )
    return {"id": request["id"], "status": "ok", "results": responses, "latency_ms": elapsed}


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
    parser.add_argument("--inference-query-mode", default="target_conditioned_text_prompt")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--disable-batch-oom-retry", action="store_true")
    args = parser.parse_args()
    if args.max_batch_size < 1:
        parser.error("max-batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        load_started = time.perf_counter()
        model = _load_detector(args)
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        batch_inference = None
        # Import only after model initialization; custom LAE-DINO mmdet APIs can
        # expose inference_detector from different namespaces.
        try:
            from mmdet.apis import inference_detector
        except ImportError as exc:
            raise RuntimeError("compatible mmdet inference_detector is unavailable") from exc
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "failure_stage": "model_init", "error": str(exc)}),
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
            if "id" not in request:
                raise ValueError("missing request field: id")
            operation = request.get("operation", "predict")
            if operation == "initialize":
                response = {
                    "id": request["id"],
                    "status": "ok",
                    "model_inventory": {
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "model_load_ms": model_load_ms,
                        "worker_pid": os.getpid(),
                        "max_batch_size": args.max_batch_size,
                    },
                }
            elif operation == "predict_batch":
                if batch_inference is None:
                    batch_inference = _build_batch_inference(model)
                response = _predict_batch(model, request, args, batch_inference)
            elif operation == "predict":
                for key in ("image", "target_phrase"):
                    if key not in request:
                        raise ValueError(f"missing request field: {key}")
                response = _predict(model, request, args, inference_detector)
            else:
                raise ValueError(f"unsupported operation: {operation}")
        except json.JSONDecodeError as exc:
            response = {"status": "failed", "failure_stage": "protocol_guard", "error": str(exc)}
        except Exception as exc:
            response = {
                "id": request.get("id") if isinstance(locals().get("request"), dict) else None,
                "status": "failed",
                "failure_stage": "proposal_generation",
                "error": str(exc),
            }
        try:
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
        finally:
            request = None
            response = None
            _release_request_resources()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
