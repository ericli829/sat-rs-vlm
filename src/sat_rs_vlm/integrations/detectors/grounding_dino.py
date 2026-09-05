"""Lazy local-only Grounding DINO Base proposal provider."""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import (
    ProposalError,
    ProposalResult,
    canonicalize_proposals,
    stable_file_identity,
)


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
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


class GroundingDinoProvider:
    """Reuse a local Grounding DINO model across many proposal requests."""

    provider_name = "grounding_dino"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        value = self.config.get("model_path") or self.config.get("model")
        if not value:
            raise ProposalError("grounding_dino requires proposal.model_path")
        self.model_path = Path(str(value)).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ProposalError(f"Grounding DINO model directory does not exist: {self.model_path}")
        self.device = str(self.config.get("device", "cuda"))
        self.dtype = str(self.config.get("dtype", "float16"))
        self.box_threshold = float(self.config.get("box_threshold", 0.3))
        self.text_threshold = float(self.config.get("text_threshold", 0.25))
        self.top_k = int(self.config.get("top_k", 100))
        nms = self.config.get("nms_threshold")
        self.nms_threshold = None if nms in (None, "", "null") else float(nms)
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._model_identity = stable_file_identity(self.model_path)
        self.model_identity = self._model_identity

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise ProposalError(
                "Grounding DINO requires transformers AutoProcessor and "
                "AutoModelForZeroShotObjectDetection in the active environment"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ProposalError(f"Grounding DINO requested {self.device}, but CUDA is unavailable")
        dtype = getattr(torch, self.dtype, None)
        kwargs: dict[str, Any] = {"local_files_only": True}
        if dtype is not None and self.device.startswith("cuda"):
            kwargs["torch_dtype"] = dtype
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self._processor = AutoProcessor.from_pretrained(
                    str(self.model_path), local_files_only=True
                )
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    str(self.model_path), **kwargs
                )
                self._model.to(self.device)
                self._model.eval()
        except Exception as exc:
            self._processor = None
            self._model = None
            raise ProposalError(f"failed to load local Grounding DINO: {exc}") from exc
        self._torch = torch

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        self._ensure_loaded()
        try:
            from PIL import Image

            image = Image.open(str(image_path)).convert("RGB")
        except Exception as exc:
            raise ProposalError(f"failed to open proposal image {image_path}: {exc}") from exc
        query = str(target_phrase).strip().lower()
        if not query:
            raise ProposalError("Grounding DINO target_phrase must not be empty")
        started = time.perf_counter()
        try:
            # Transformers 4.57+ documents a structured single-image label
            # batch.  Keep the query as one label instead of inventing
            # punctuation/caption rules.
            text_labels = [[query]]
            inputs = self._processor(images=image, text=text_labels, return_tensors="pt")
            inputs = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(image.height, image.width)],
                text_labels=text_labels,
            )[0]
            boxes = processed.get("boxes", [])
            scores = processed.get("scores", [])
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().tolist()
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().tolist()
            boxes, scores, stats = canonicalize_proposals(
                boxes,
                scores,
                image_width=image.width,
                image_height=image.height,
                coordinate_mode="pixel",
                # NMS must see the complete thresholded/sorted proposal set;
                # final top-k is applied only after suppression.
                top_k=None,
            )
            if self.nms_threshold is not None:
                boxes, scores = _nms(boxes, scores, self.nms_threshold)
            boxes, scores = boxes[: self.top_k], scores[: self.top_k]
        except Exception as exc:
            raise ProposalError(f"Grounding DINO inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ProposalResult(
            boxes_xyxy=boxes,
            scores=scores,
            latency_ms=latency_ms,
            provider=self.provider_name,
            model_id=str(self.model_path),
            metadata={
                "schema_version": "grounding-dino-provider-v1",
                "target_phrase": query,
                "text_labels": text_labels,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "top_k": self.top_k,
                "nms_threshold": self.nms_threshold,
                "coordinate_mode": "absolute_pixel_xyxy",
                "image_width": image.width,
                "image_height": image.height,
                "validation": stats,
                "model_identity": self._model_identity,
            },
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None
