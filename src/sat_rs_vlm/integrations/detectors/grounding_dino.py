"""Lazy local-only Grounding DINO Base proposal provider."""

from __future__ import annotations

import contextlib
import io
import math
import tempfile
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
from .instance_postprocess import conservative_nms_indices, postprocess_instances


def _nms(
    boxes: list[list[float]], scores: list[float], threshold: float
) -> tuple[list[list[float]], list[float]]:
    try:
        keep = conservative_nms_indices(boxes, scores, threshold)
    except ValueError as exc:
        raise ProposalError(str(exc)) from exc
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

    def predict_raw(self, image_path: Path, target_phrase: str) -> ProposalResult:
        """Return canonical proposals before NMS/refinement/top-k.

        TiledProposalProvider uses this method so overlapping tiles are fused
        once in original-image coordinates, rather than independently per tile.
        """

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
                "candidate_truncated": False,
                "candidate_pool_complete": stats["invalid_count"] == 0,
            },
        )

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        started = time.perf_counter()
        raw = self.predict_raw(image_path, target_phrase)
        nms_iou = (
            self.nms_threshold
            if self.nms_threshold is not None
            else float(self.config.get("instance_nms_iou", 0.6))
        )
        padding = float(self.config.get("instance_refinement_padding_ratio", 0.12))
        max_regions = int(self.config.get("instance_max_refinement_regions", 4))

        def refine(parent: list[float]) -> tuple[list[list[float]], list[float]]:
            from PIL import Image

            with Image.open(str(image_path)) as source:
                image = source.convert("RGB")
            try:
                width, height = image.size
                pad_x = max(8, round((parent[2] - parent[0]) * padding))
                pad_y = max(8, round((parent[3] - parent[1]) * padding))
                crop_box = (
                    max(0, math.floor(parent[0] - pad_x)),
                    max(0, math.floor(parent[1] - pad_y)),
                    min(width, math.ceil(parent[2] + pad_x)),
                    min(height, math.ceil(parent[3] + pad_y)),
                )
                with tempfile.TemporaryDirectory(prefix="dino_instance_refine_") as tmp:
                    crop_path = Path(tmp) / "search_region.png"
                    crop = image.crop(crop_box)
                    try:
                        crop.save(crop_path)
                    finally:
                        crop.close()
                    refined = self.predict_raw(crop_path, target_phrase)
                dx, dy = crop_box[0], crop_box[1]
                mapped = [
                    [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
                    for box in refined.boxes_xyxy
                ]
                return mapped, refined.scores
            finally:
                image.close()

        width = int(raw.metadata["image_width"])
        height = int(raw.metadata["image_height"])
        boxes, scores, instance_trace = postprocess_instances(
            raw.boxes_xyxy,
            raw.scores,
            nms_iou=float(nms_iou),
            reference_areas=[float(width * height)] * len(raw.boxes_xyxy),
            refine=refine,
            max_refinement_regions=max_regions,
            large_box_area_fraction=float(
                self.config.get("instance_large_box_area_fraction", 0.20)
            ),
        )
        candidate_count_before_top_k = len(boxes)
        boxes, scores = boxes[: self.top_k], scores[: self.top_k]
        metadata = dict(raw.metadata)
        metadata.update(
            {
                "instance_postprocess": instance_trace,
                "instance_search_regions": instance_trace["aggregate_regions"],
                "unresolved_instance_regions": instance_trace["unresolved_regions"],
                "candidate_truncated": candidate_count_before_top_k > len(boxes),
                "candidate_pool_complete": (
                    raw.metadata.get("candidate_pool_complete") is not False
                    and not instance_trace["unresolved_regions"]
                    and candidate_count_before_top_k == len(boxes)
                ),
                "postprocess_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return ProposalResult(
            boxes_xyxy=boxes,
            scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=raw.provider,
            model_id=raw.model_id,
            metadata=metadata,
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None
