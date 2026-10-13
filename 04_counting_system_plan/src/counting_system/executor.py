"""COUNT(image|entities, target, entire) -> CountResult / ScalarInt。

对齐 feature/vlm-semantic-alignment：
- 输入角色恰好其一：image=ImageRef|Region 或 entities=EntitySet|SelectResult
- COUNT(EntitySet/空 SELECT) = cardinality，不再检测
- COUNT(image/Region) = tiled 检测 + 全局坐标去重
- TaskGraph 对外只暴露 ScalarInt
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from sat_rs_vlm.integrations.detectors.instance_postprocess import postprocess_instances
from sat_rs_vlm.integrations.detectors.strategies import resolve_detection_strategy

from .contracts import resolve_count_scope, validate_count_inputs
from .detector.base import DetectionRequest
from .detector.lae_dino import build_detector
from .fusion import duplicate_rate, fuse_detections
from .image_ops import crop_tile, ensure_size, load_image
from .paths import load_config
from .retriever.fake import FakeRetriever
from .runtime import (
    CountResult,
    Detection,
    DetectionSet,
    EntitySet,
    ImageRef,
    Region,
    ScalarInt,
    SelectResult,
    detections_from_entities,
)
from .target import TargetSpec, build_target
from .tiling import Tile, plan_tiles, scope_from_input
from .trace import TraceWriter


@dataclass(frozen=True)
class CountParams:
    target: TargetSpec
    entire: bool

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target.to_params(), "entire": self.entire}


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
        self.provider_name = "count"

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

    def execute(
        self,
        inputs: dict[str, ImageRef | Region | EntitySet | SelectResult],
        params: CountParams | dict[str, Any] | None = None,
        *,
        source_scale: int | None = None,
        score_threshold: float | None = None,
        trace: TraceWriter | None = None,
    ) -> ScalarInt:
        """TaskGraph 节点入口：返回权威 ScalarInt，不回灌 VLM 再数一遍。"""
        result = self.run(
            inputs,
            params,
            source_scale=source_scale,
            score_threshold=score_threshold,
            trace=trace,
        )
        return result.to_scalar()

    def run(
        self,
        inputs: dict[str, ImageRef | Region | EntitySet | SelectResult],
        params: CountParams | dict[str, Any] | None = None,
        *,
        source_scale: int | None = None,
        score_threshold: float | None = None,
        trace: TraceWriter | None = None,
    ) -> CountResult:
        validate_count_inputs(inputs)
        parsed = self._parse_params(params, inputs)
        scope = resolve_count_scope(inputs)
        t0 = time.perf_counter()
        if isinstance(scope, EntitySet):
            result = self._count_entity_set(scope, parsed.target)
        else:
            entire = parsed.entire if isinstance(scope, ImageRef) else False
            result = self._count_visual(
                scope,
                parsed.target,
                entire=entire,
                source_scale=source_scale,
                score_threshold=score_threshold,
            )
        result.provenance["latency_sec"] = time.perf_counter() - t0
        result.provenance["count_params"] = parsed.to_dict()
        if trace:
            trace.record(result)
        return result

    def __call__(
        self,
        inputs: ImageRef | Region | EntitySet | SelectResult | dict[str, Any],
        target: TargetSpec | str | None = None,
        *,
        entire: bool | None = None,
        source_scale: int | None = None,
        score_threshold: float | None = None,
        trace: TraceWriter | None = None,
        params: CountParams | dict[str, Any] | None = None,
    ) -> CountResult:
        if isinstance(inputs, dict):
            return self.run(
                inputs,
                params or self._params_from_target(target, entire, inputs),
                source_scale=source_scale,
                score_threshold=score_threshold,
                trace=trace,
            )
        if isinstance(inputs, EntitySet | SelectResult):
            payload = {"entities": inputs}
        else:
            payload = {"image": inputs}
        return self.run(
            payload,
            params or self._params_from_target(target, entire, payload),
            source_scale=source_scale,
            score_threshold=score_threshold,
            trace=trace,
        )

    def _params_from_target(
        self,
        target: TargetSpec | str | None,
        entire: bool | None,
        inputs: dict[str, Any],
    ) -> CountParams:
        spec = target if isinstance(target, TargetSpec) else build_target(str(target or "object"))
        if entire is None:
            entire = bool(self.config.get("count", {}).get("entire", True))
        if "image" in inputs and isinstance(inputs["image"], Region):
            entire = False
        if "entities" in inputs:
            entire = False
        return CountParams(target=spec, entire=bool(entire))

    def _parse_params(
        self,
        params: CountParams | dict[str, Any] | None,
        inputs: dict[str, Any],
    ) -> CountParams:
        if isinstance(params, CountParams):
            if isinstance(inputs.get("image"), Region):
                return CountParams(target=params.target, entire=False)
            return params
        payload = dict(params or {})
        raw_target = payload.get("target")
        if isinstance(raw_target, TargetSpec):
            spec = raw_target
        elif isinstance(raw_target, dict):
            spec = build_target(str(raw_target.get("category") or "object"))
            if raw_target.get("attributes"):
                spec.attributes = dict(raw_target["attributes"])
        elif raw_target:
            spec = build_target(str(raw_target))
        else:
            spec = build_target("object")
        entire = payload.get("entire")
        if entire is None:
            entire = bool(self.config.get("count", {}).get("entire", True))
        if isinstance(inputs.get("image"), Region) or "entities" in inputs:
            entire = False
        return CountParams(target=spec, entire=bool(entire))

    def _count_entity_set(self, entities: EntitySet, spec: TargetSpec) -> CountResult:
        dets = detections_from_entities(entities)
        return CountResult(
            count=len(entities.entities),
            detections=dets,
            provenance={
                "mode": "entityset",
                "target": spec.category,
                "phrase": spec.phrase(),
                "redetect": False,
                "detector_calls": 0,
                "provider": "cardinality",
                "source": "EntitySet",
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
        strategy = resolve_detection_strategy(
            spec.category, self.config.get("detection_strategies")
        )
        if strategy["name"] != "legacy":
            gate_cfg = {"enabled": False}
        image, region = _split_visual(inputs)
        pil = load_image(image)
        image = ensure_size(image, pil)
        if region is not None:
            region = Region(image, region.bbox_xyxy_global, provenance=dict(region.provenance))
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
        tile_diagnostics = []
        invalid_global_candidates = 0
        detector_error = None

        def requests():
            for tile in gated:
                crop = crop_tile(pil, tile)
                try:
                    yield DetectionRequest(
                        image=crop,
                        target=spec,
                        tile=tile,
                        score_threshold=float(
                            strategy.get(
                                "score_threshold",
                                (self.config.get("detector") or {}).get("score_threshold") or 0.0,
                            )
                        ),
                        texts=spec.texts(),
                    )
                finally:
                    crop.close()

        detect_many = getattr(detector, "detect_many", None)
        pending_requests = requests()
        try:
            # The bridge owns CPU prefetch; this loop never increases inference concurrency.
            responses = (
                detect_many(pending_requests)
                if callable(detect_many)
                else (detector.detect(request) for request in pending_requests)
            )
            try:
                for response in responses:
                    for det in response.detections:
                        a, b, c, d = det.bbox_xyxy_global
                        box = (
                            max(a, scope[0]),
                            max(b, scope[1]),
                            min(c, scope[2]),
                            min(d, scope[3]),
                        )
                        if box[0] >= box[2] or box[1] >= box[3]:
                            invalid_global_candidates += 1
                            continue
                        if box != det.bbox_xyxy_global:
                            det.provenance["bbox_before_scope_clip"] = list(det.bbox_xyxy_global)
                            det.bbox_xyxy_global = box
                        raw.append(det)
                    tile_diagnostics.append(dict(response.extra.get("proposal_metadata", {})))
                    calls += 1
            except Exception as exc:
                if strategy["name"] == "legacy":
                    raise
                detector_error = f"{type(exc).__name__}: {exc}"
            finally:
                close_responses = getattr(responses, "close", None)
                if callable(close_responses):
                    close_responses()
        finally:
            pending_requests.close()
            pil.close()
        thr = float(
            score_threshold
            if score_threshold is not None
            else strategy.get("score_threshold", count_cfg.get("score_threshold", 0.2))
        )
        fused, stats = fuse_detections(
            raw,
            tiles,
            score_threshold=thr,
            nms_iou=float(count_cfg.get("nms_iou", 0.4)),
            cross_iou=float(count_cfg.get("cross_scale_iou", 0.5)),
        )
        instance_trace = None
        refinement_calls = 0
        if strategy["name"] != "legacy":
            by_tile = {tile.tile_id: tile for tile in tiles}

            def refine_region(parent):
                nonlocal refinement_calls
                pad_x = max(8, round((parent[2] - parent[0]) * 0.12))
                pad_y = max(8, round((parent[3] - parent[1]) * 0.12))
                crop_box = (
                    max(scope[0], math.floor(parent[0] - pad_x)),
                    max(scope[1], math.floor(parent[1] - pad_y)),
                    min(scope[2], math.ceil(parent[2] + pad_x)),
                    min(scope[3], math.ceil(parent[3] + pad_y)),
                )
                tile = Tile(
                    tile_id=f"instance_refinement:{refinement_calls}",
                    scale_id=strategy["name"],
                    image=image,
                    crop_xyxy=crop_box,
                    core_xyxy=crop_box,
                    detector_input=0,
                    scope_xyxy=scope,
                    spatial_resize_enabled=bool(strategy["spatial_resize_enabled"]),
                )
                with load_image(image) as refinement_image:
                    crop = crop_tile(refinement_image, tile)
                    try:
                        response = detector.detect(
                            DetectionRequest(
                                image=crop,
                                target=spec,
                                tile=tile,
                                score_threshold=thr,
                                texts=spec.texts(),
                            )
                        )
                    finally:
                        crop.close()
                refinement_calls += 1
                tile_diagnostics.append(dict(response.extra.get("proposal_metadata", {})))
                return (
                    [list(det.bbox_xyxy_global) for det in response.detections],
                    [det.score for det in response.detections],
                )

            boxes, scores, instance_trace = postprocess_instances(
                [list(det.bbox_xyxy_global) for det in fused],
                [det.score for det in fused],
                nms_iou=max(0.6, float(count_cfg.get("nms_iou", 0.4))),
                source_ids=list(range(len(fused))),
                reference_areas=[
                    float(by_tile[det.tile_id].crop_size[0] * by_tile[det.tile_id].crop_size[1])
                    if det.tile_id in by_tile
                    else 0.0
                    for det in fused
                ],
                refine=refine_region,
                max_refinement_regions=int(strategy["max_refinement_regions"]),
                large_box_area_fraction=float(strategy["large_box_area_fraction"]),
            )
            resolved = []
            source_occurrences = {}
            for box, score, source_id in zip(
                boxes, scores, instance_trace["final_source_ids"], strict=False
            ):
                original = fused[int(source_id)]
                occurrence = source_occurrences.get(source_id, 0)
                source_occurrences[source_id] = occurrence + 1
                provenance = dict(original.provenance)
                if tuple(box) != original.bbox_xyxy_global or occurrence:
                    provenance["source_candidate_id"] = provenance.get("candidate_id")
                    provenance["candidate_id"] = (
                        f"{provenance.get('candidate_id')}:refined:{occurrence + 1}"
                    )
                resolved.append(
                    Detection(
                        tuple(box),
                        score,
                        original.label,
                        original.tile_id,
                        original.scale_id,
                        provenance=provenance,
                    )
                )
            fused = resolved
            stats["after_instance_postprocess"] = len(fused)
        pool_complete = (
            calls == len(gated) == len(tiles)
            and invalid_global_candidates == 0
            and all(
                d.get("candidate_pool_complete") is not False and not d.get("candidate_truncated")
                for d in tile_diagnostics
            )
        )
        instance_complete = all(
            d.get("instance_resolution_complete") is not False for d in tile_diagnostics
        )
        if instance_trace is not None:
            instance_complete = instance_complete and instance_trace["instance_resolution_complete"]
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
                "target": spec.category,
                "phrase": spec.phrase(),
                "prompt": spec.texts(),
                "tiny": spec.tiny,
                "scope": list(scope),
                "source_scale": source_scale
                or (self.config.get("scale") or {}).get("default_source_scale"),
                "tiles_planned": len(tiles),
                "tiles_run": len(gated),
                "detector": getattr(detector, "impl_name", getattr(detector, "name", "unknown")),
                "provider": getattr(detector, "impl_name", getattr(detector, "name", "unknown")),
                "detector_calls": calls + refinement_calls,
                "instance_refinement_calls": refinement_calls,
                "raw_proposals": [asdict(d) for d in raw]
                if count_cfg.get("keep_raw_proposals", True)
                else len(raw),
                "raw_count": len(raw),
                "fusion": stats,
                "detection_strategy": strategy,
                "tile_diagnostics": tile_diagnostics,
                "invalid_global_candidate_count": invalid_global_candidates,
                "detector_error": detector_error,
                "candidate_pool_complete": pool_complete,
                "candidate_truncated": any(d.get("candidate_truncated") for d in tile_diagnostics),
                "instance_resolution_complete": instance_complete,
                "instance_postprocess": instance_trace,
                "unresolved_instance_regions": (
                    instance_trace["unresolved_regions"]
                    if instance_trace
                    else [
                        region
                        for d in tile_diagnostics
                        for region in d.get("unresolved_instance_regions", [])
                    ]
                ),
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
            text = f"a remote sensing image with {spec.phrase()}"
            score = float(retriever.score(crop, text))
            scores.append({"tile_id": tile.tile_id, "score": score})
            if score >= threshold:
                keep_ids.add(tile.tile_id)
                survivors.append(tile)
        if not survivors:
            trace.update(
                {"survivors": len(tiles), "dropped": 0, "scores": scores, "fallback": "all"}
            )
            return tiles, trace
        if others:
            survivors.extend(others)
        dropped = len(scan) - len(keep_ids)
        trace.update(
            {
                "survivors": len(survivors),
                "dropped": dropped,
                "scores": scores,
                "threshold": threshold,
            }
        )
        return survivors, trace


def _split_visual(inputs: ImageRef | Region) -> tuple[ImageRef, Region | None]:
    if isinstance(inputs, Region):
        return inputs.image, inputs
    return inputs, None


def count(
    inputs: ImageRef | Region | EntitySet | SelectResult | dict[str, Any],
    target: TargetSpec | str | None = None,
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
