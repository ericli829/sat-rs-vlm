"""COUNT(target, entire) -> CountResult / ScalarInt。"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .detector.base import DetectionRequest
from .detector.lae_dino import build_detector
from .fusion import duplicate_rate, fuse_detections
from .image_ops import crop_tile, ensure_size, load_image
from .paths import load_config
from .retriever.fake import FakeRetriever
from .runtime import CountResult, Detection, DetectionSet, EntitySet, ImageRef, Region, ScalarInt, detections_from_entities
from .target import TargetSpec, build_target
from .tiling import Tile, plan_tiles, scope_from_input
from .trace import TraceWriter


class CountExecutor:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        detector=None,
        retriever=None,
    ):
        self.config = load_config(config)
        self.detector = detector
        self.retriever = retriever
        self._owns_detector = detector is None
        self._owns_retriever = retriever is None

    def close(self) -> None:
        if self._owns_detector and self.detector is not None:
            close = getattr(self.detector, "close", None)
            if close:
                close()
        if self._owns_retriever and self.retriever is not None:
            close = getattr(self.retriever, "close", None)
            if close:
                close()

    def __enter__(self) -> CountExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_detector(self):
        if self.detector is None:
            self.detector = build_detector(self.config)
        return self.detector

    def _ensure_retriever(self):
        if self.retriever is None:
            gate = self.config.get("gate") or {}
            if not gate.get("enabled"):
                self.retriever = FakeRetriever()
            else:
                from .retriever.georsclip import build_retriever

                try:
                    self.retriever = build_retriever(self.config)
                except Exception:
                    self.retriever = FakeRetriever()
        return self.retriever

    def __call__(
        self,
        inputs: ImageRef | Region | EntitySet,
        target: TargetSpec | str,
        *,
        entire: bool | None = None,
        source_scale: int | None = None,
        score_threshold: float | None = None,
        trace: TraceWriter | None = None,
    ) -> CountResult:
        t0 = time.perf_counter()
        spec = target if isinstance(target, TargetSpec) else build_target(str(target))
        if entire is None:
            entire = bool(self.config.get("count", {}).get("entire", True))
        if isinstance(inputs, EntitySet):
            result = self._count_entity_set(inputs, spec)
            result.provenance["latency_sec"] = time.perf_counter() - t0
            if trace:
                trace.record(result)
            return result
        result = self._count_visual(
            inputs,
            spec,
            entire=bool(entire),
            source_scale=source_scale,
            score_threshold=score_threshold,
        )
        result.provenance["latency_sec"] = time.perf_counter() - t0
        if trace:
            trace.record(result)
        return result

    def _count_entity_set(self, entities: EntitySet, spec: TargetSpec) -> CountResult:
        dets = detections_from_entities(entities)
        return CountResult(
            count=len(entities),
            detections=dets,
            provenance={
                "mode": "entityset",
                "target": spec.name,
                "redetect": False,
                "detector_calls": 0,
            },
        )

    def _count_visual(
        self,
        inputs: ImageRef | Region,
        spec: TargetSpec,
        *,
        entire: bool,
        source_scale: int | None,
        score_threshold: float | None,
    ) -> CountResult:
        count_cfg = self.config.get("count") or {}
        gate_cfg = self.config.get("gate") or {}
        image, region = _split_visual(inputs)
        pil = load_image(image)
        ensure_size(image, pil)
        _image, scope = scope_from_input(image, region)
        tiles = plan_tiles(
            image,
            scope,
            spec,
            self.config,
            entire=entire,
            source_scale=source_scale,
        )
        gated, gate_trace = self._maybe_gate(pil, tiles, spec, entire=entire, gate_cfg=gate_cfg)
        detector = self._ensure_detector()
        raw: list[Detection] = []
        calls = 0
        for tile in gated:
            crop = crop_tile(pil, tile)
            request = DetectionRequest(
                image=crop,
                target=spec,
                tile=tile,
                score_threshold=float((self.config.get("detector") or {}).get("score_threshold") or 0.0),
                texts=spec.texts(),
            )
            response = detector.detect(request)
            raw.extend(response.detections)
            calls += 1
        thr = float(score_threshold if score_threshold is not None else count_cfg.get("score_threshold", 0.2))
        fused, stats = fuse_detections(
            raw,
            tiles,
            score_threshold=thr,
            nms_iou=float(count_cfg.get("nms_iou", 0.5)),
            cross_iou=float(count_cfg.get("cross_scale_iou", 0.5)),
        )
        for i, det in enumerate(fused):
            det.provenance = dict(det.provenance)
            det.provenance["kept"] = True
            det.provenance["index"] = i
        result = CountResult(
            count=len(fused),
            detections=DetectionSet(fused),
            provenance={
                "mode": "visual",
                "entire": entire,
                "target": spec.name,
                "prompt": spec.texts(),
                "tiny": spec.tiny,
                "scope": list(scope),
                "source_scale": source_scale or (self.config.get("scale") or {}).get("default_source_scale"),
                "tiles_planned": len(tiles),
                "tiles_run": len(gated),
                "detector": getattr(detector, "impl_name", getattr(detector, "name", "unknown")),
                "detector_calls": calls,
                "raw_proposals": [asdict(d) for d in raw] if count_cfg.get("keep_raw_proposals", True) else len(raw),
                "raw_count": len(raw),
                "fusion": stats,
                "gate": gate_trace,
                "duplicate_rate": duplicate_rate(fused),
            },
        )
        return result

    def _maybe_gate(
        self,
        pil,
        tiles: list[Tile],
        spec: TargetSpec,
        *,
        entire: bool,
        gate_cfg: dict[str, Any],
    ) -> tuple[list[Tile], dict[str, Any]]:
        enabled = bool(gate_cfg.get("enabled")) and entire
        trace = {"enabled": enabled, "survivors": len(tiles), "dropped": 0, "scores": []}
        if not enabled:
            return tiles, trace
        retriever = self._ensure_retriever()
        threshold = float(gate_cfg.get("threshold", 0.12))
        survivors: list[Tile] = []
        scores: list[dict[str, Any]] = []
        native = [t for t in tiles if t.scale_id == "native"]
        others = [t for t in tiles if t.scale_id != "native"]
        keep_ids: set[str] = set()
        scan = native or tiles
        for tile in scan:
            crop = pil.crop(tuple(int(round(v)) for v in tile.crop_xyxy))
            text = f"a remote sensing image with {spec.name}"
            score = float(retriever.score(crop, text))
            scores.append({"tile_id": tile.tile_id, "score": score})
            if score >= threshold:
                keep_ids.add(tile.tile_id)
                survivors.append(tile)
        if not survivors:
            # recall first：全灭则退回全量，避免 gate 误杀
            trace.update({"survivors": len(tiles), "dropped": 0, "scores": scores, "fallback": "all"})
            return tiles, trace
        if others:
            survivors.extend(others)
        dropped = len(scan) - len(keep_ids)
        trace.update({"survivors": len(survivors), "dropped": dropped, "scores": scores, "threshold": threshold})
        return survivors, trace


def _split_visual(inputs: ImageRef | Region) -> tuple[ImageRef, Region | None]:
    if isinstance(inputs, Region):
        return inputs.image, inputs
    return inputs, None


def count(
    inputs: ImageRef | Region | EntitySet,
    target: TargetSpec | str,
    *,
    entire: bool | None = None,
    config: dict[str, Any] | None = None,
    detector=None,
    **kwargs: Any,
) -> CountResult:
    executor = CountExecutor(config, detector=detector)
    try:
        return executor(inputs, target, entire=entire, **kwargs)
    finally:
        if detector is None:
            executor.close()


def as_scalar(result: CountResult) -> ScalarInt:
    return result.to_scalar()
