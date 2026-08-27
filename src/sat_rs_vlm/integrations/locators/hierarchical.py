"""Deterministic query-aware hierarchical beam locator."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sat_rs_vlm.integrations.detectors.protocol import (
    ProposalProvider,
    ProposalResult,
)
from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider
from sat_rs_vlm.semantics import QueryParser, TaskSpec

from .beam import StopPolicyConfig, adaptive_beam_select, evaluate_stop
from .fusion import RegionFusion
from .geometry import bbox_area, clamp_bbox, expand_with_halo, subdivide_core
from .router import TaskRouter
from .scoring import (
    CompositeRegionScorer,
    DetectorRegionScorer,
    RetrievalRegionScorer,
    ScoreBatch,
    SpatialRegionScorer,
)
from .types import BBox, LocatorError, LocatorResult, SearchPlan, SearchRegion


@dataclass(frozen=True)
class HierarchicalSearchConfig:
    grid_size: int = 3
    halo_ratio: float = 0.15
    temperature: float = 1.0
    cumulative_mass: float = 0.9
    max_beam: int = 4

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> HierarchicalSearchConfig:
        values = dict(config or {})
        result = cls(
            grid_size=int(values.get("grid_size", cls.grid_size)),
            halo_ratio=float(values.get("halo_ratio", cls.halo_ratio)),
            temperature=float(values.get("temperature", cls.temperature)),
            cumulative_mass=float(values.get("cumulative_mass", cls.cumulative_mass)),
            max_beam=int(values.get("max_beam", cls.max_beam)),
        )
        if result.grid_size < 2 or result.max_beam < 1:
            raise LocatorError("search grid_size/max_beam values are invalid")
        if result.halo_ratio < 0.0 or result.temperature <= 0.0:
            raise LocatorError("search halo_ratio/temperature values are invalid")
        if not 0.0 < result.cumulative_mass <= 1.0:
            raise LocatorError("search cumulative_mass must be in (0, 1]")
        return result


class HierarchicalLocator:
    provider_name = "uhr_hierarchical"

    def __init__(
        self,
        *,
        parser: QueryParser,
        router: TaskRouter,
        config: Mapping[str, Any],
        detector_provider: ProposalProvider | None = None,
        retriever_provider: RetrieverProvider | None = None,
    ) -> None:
        self.parser = parser
        self.router = router
        self.config = dict(config)
        self.search_config = HierarchicalSearchConfig.from_mapping(
            self.config.get("search", {})
        )
        self.stop_config = StopPolicyConfig.from_mapping(self.config.get("search", {}))
        scorer_config = dict(self.config.get("scorers", {}))
        self.detector_enabled = bool(
            dict(scorer_config.get("detector", {})).get("enabled", True)
        )
        self.retrieval_enabled = bool(
            dict(scorer_config.get("retrieval", {})).get("enabled", True)
        )
        self.spatial_enabled = bool(
            dict(scorer_config.get("spatial", {})).get("enabled", True)
        )
        self.detector_provider = detector_provider
        self.retriever_provider = retriever_provider
        self.detector_scorer = DetectorRegionScorer()
        self.retrieval_scorer = RetrievalRegionScorer(retriever_provider)
        self.spatial_scorer = SpatialRegionScorer()
        self.composite_scorer = CompositeRegionScorer(scorer_config.get("weights", {}))
        self.fusion = RegionFusion(self.config.get("fusion", {}))
        self.fail_on_provider_error = bool(self.config.get("fail_on_provider_error", True))

    def _collect_proposals(
        self,
        image_path: Path,
        plan: SearchPlan,
        warnings: list[str],
    ) -> tuple[ProposalResult | None, float]:
        if not plan.use_detector or not self.detector_enabled:
            return None, 0.0
        if self.detector_provider is None:
            warnings.append("detector_provider_unavailable")
            return None, 0.0
        boxes: list[list[float]] = []
        scores: list[float] = []
        sources: list[dict[str, Any]] = []
        latency = 0.0
        for phrase in plan.target_phrases:
            try:
                result = self.detector_provider.predict(image_path, phrase)
            except Exception as exc:
                if self.fail_on_provider_error:
                    raise
                warnings.append(f"detector_error:{type(exc).__name__}:{exc}")
                continue
            boxes.extend(result.boxes_xyxy)
            scores.extend(result.scores)
            latency += result.latency_ms
            sources.append(
                {
                    "target_phrase": phrase,
                    "provider": result.provider,
                    "model_id": result.model_id,
                    "latency_ms": result.latency_ms,
                    "metadata": dict(result.metadata),
                }
            )
        if not sources:
            return None, latency
        return (
            ProposalResult(
                boxes_xyxy=boxes,
                scores=scores,
                latency_ms=latency,
                provider=self.detector_provider.provider_name,
                model_id=str(getattr(self.detector_provider, "model_id", "unknown")),
                metadata={"queries": sources, "coordinate_mode": "absolute_pixel_xyxy"},
            ),
            latency,
        )

    def _score_children(
        self,
        *,
        image_path: Path,
        task: TaskSpec,
        plan: SearchPlan,
        children: Sequence[SearchRegion],
        proposals: ProposalResult | None,
        image_width: int,
        image_height: int,
        parent_score: float,
        warnings: list[str],
    ) -> tuple[tuple[float, ...], tuple[dict[str, Any], ...], float, tuple[str, ...]]:
        batches: list[ScoreBatch] = []
        if plan.use_detector and self.detector_enabled:
            batches.append(self.detector_scorer.score(task, children, proposals))
        else:
            batches.append(
                ScoreBatch.unavailable("detector", len(children), "disabled_by_plan_or_config")
            )
        retrieval_latency = 0.0
        if plan.use_retrieval and self.retrieval_enabled:
            try:
                retrieval_batch = self.retrieval_scorer.score(
                    image_path,
                    task.raw_question,
                    children,
                )
            except Exception as exc:
                if self.fail_on_provider_error:
                    raise
                warnings.append(f"retriever_error:{type(exc).__name__}:{exc}")
                retrieval_batch = ScoreBatch.unavailable(
                    "retrieval", len(children), "retriever_failure"
                )
            batches.append(retrieval_batch)
            if retrieval_batch.available and retrieval_batch.metadata:
                retrieval_latency = float(retrieval_batch.metadata[0].get("latency_ms", 0.0))
        else:
            batches.append(
                ScoreBatch.unavailable("retrieval", len(children), "disabled_by_plan_or_config")
            )
        if plan.use_spatial and self.spatial_enabled:
            batches.append(
                self.spatial_scorer.score(task, children, image_width, image_height)
            )
        else:
            batches.append(
                ScoreBatch.unavailable("spatial", len(children), "disabled_by_plan_or_config")
            )
        composite = self.composite_scorer.score(
            children,
            batches,
            parent_score=parent_score,
        )
        return (
            composite.scores,
            composite.components,
            retrieval_latency,
            composite.active_scorers,
        )

    @staticmethod
    def _bypass_result(
        *,
        task: TaskSpec,
        plan: SearchPlan,
        image_width: int,
        image_height: int,
        parser_latency: float,
        started: float,
    ) -> LocatorResult:
        box = (
            clamp_bbox(task.given_bbox, image_width, image_height)
            if task.given_bbox is not None
            else (0.0, 0.0, float(image_width), float(image_height))
        )
        total = (time.perf_counter() - started) * 1000.0
        return LocatorResult(
            regions_xyxy=(box,),
            scores=(1.0,),
            task_spec=task,
            search_plan=plan,
            search_trace=(
                {
                    "region_id": "direct",
                    "selected": True,
                    "core_xyxy": list(box),
                    "view_xyxy": list(box),
                    "reason": plan.route,
                },
            ),
            inspected_area_ratio=bbox_area(box) / (image_width * image_height),
            depth_reached=0,
            latency_ms={
                "parser": parser_latency,
                "detector": 0.0,
                "retrieval": 0.0,
                "search": 0.0,
                "fusion": 0.0,
                "total": total,
            },
            provider_provenance={"locator": "uhr_hierarchical", "bypass": plan.route},
            warnings=tuple(dict.fromkeys((*task.warnings, *plan.warnings))),
            region_details=(
                {
                    "region_id": "direct",
                    "core_xyxy": list(box),
                    "view_xyxy": list(box),
                    "score": 1.0,
                    "coordinate_mode": "absolute_original_pixel_xyxy",
                },
            ),
        )

    def locate(self, image_path: Path, query: str | TaskSpec) -> LocatorResult:
        started = time.perf_counter()
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise LocatorError(f"locator image does not exist: {resolved_image}")
        parser_started = time.perf_counter()
        task = query if isinstance(query, TaskSpec) else self.parser.parse(query)
        parser_latency = (time.perf_counter() - parser_started) * 1000.0
        plan = self.router.route(task)
        with Image.open(resolved_image) as image:
            image_width, image_height = image.size
        if plan.bypass_locator:
            return self._bypass_result(
                task=task,
                plan=plan,
                image_width=image_width,
                image_height=image_height,
                parser_latency=parser_latency,
                started=started,
            )

        warnings = list(dict.fromkeys((*task.warnings, *plan.warnings)))
        proposals, detector_latency = self._collect_proposals(
            resolved_image,
            plan,
            warnings,
        )
        root_box: BBox = (0.0, 0.0, float(image_width), float(image_height))
        root = SearchRegion(
            region_id="root",
            parent_id=None,
            depth=0,
            core_xyxy=root_box,
            view_xyxy=root_box,
            score=0.0,
            metadata={"coordinate_mode": "absolute_original_pixel_xyxy"},
        )
        frontier = [root]
        leaves: list[SearchRegion] = []
        trace: list[dict[str, Any]] = []
        evaluated_regions = 0
        cumulative_inspected_area = 0.0
        image_area = float(image_width * image_height)
        retrieval_latency = 0.0
        search_started = time.perf_counter()

        while frontier:
            next_frontier: list[SearchRegion] = []
            for parent in frontier:
                remaining_budget = self.stop_config.max_regions - evaluated_regions
                child_count = self.search_config.grid_size**2
                if remaining_budget < child_count:
                    parent.metadata["stop_reasons"] = ["max_regions_before_subdivision"]
                    leaves.append(parent)
                    continue
                parent_boxes = subdivide_core(parent.core_xyxy, self.search_config.grid_size)
                children = [
                    SearchRegion(
                        region_id=f"{parent.region_id}.{index}",
                        parent_id=parent.region_id,
                        depth=parent.depth + 1,
                        core_xyxy=box,
                        view_xyxy=expand_with_halo(
                            box,
                            self.search_config.halo_ratio,
                            image_width,
                            image_height,
                        ),
                        metadata={"coordinate_mode": "absolute_original_pixel_xyxy"},
                    )
                    for index, box in enumerate(parent_boxes)
                ]
                parent.children.extend(child.region_id for child in children)
                evaluated_regions += len(children)
                cumulative_inspected_area += sum(
                    bbox_area(child.view_xyxy) for child in children
                )
                inspected_ratio = cumulative_inspected_area / image_area
                scores, components, batch_latency, active_scorers = self._score_children(
                    image_path=resolved_image,
                    task=task,
                    plan=plan,
                    children=children,
                    proposals=proposals,
                    image_width=image_width,
                    image_height=image_height,
                    parent_score=parent.score,
                    warnings=warnings,
                )
                retrieval_latency += batch_latency
                for child, score, detail in zip(children, scores, components, strict=True):
                    child.score = score
                    child.score_components = detail
                selection = adaptive_beam_select(
                    children,
                    scores,
                    temperature=self.search_config.temperature,
                    cumulative_mass=self.search_config.cumulative_mass,
                    max_beam=self.search_config.max_beam,
                    redundancy_weight=self.composite_scorer.weights["redundancy"],
                )
                selected_set = set(selection.selected_indices)
                posterior_max = max(selection.probabilities)
                for index, child in enumerate(children):
                    score_gain = child.score - parent.score if parent.depth > 0 else None
                    decision = evaluate_stop(
                        child,
                        self.stop_config,
                        evaluated_regions=evaluated_regions,
                        inspected_area_ratio=inspected_ratio,
                        score_gain=score_gain,
                        posterior_max=(
                            posterior_max
                            if index in selected_set
                            and selection.probabilities[index] == posterior_max
                            else None
                        ),
                    )
                    selected = index in selected_set
                    child.metadata.update(
                        {
                            "selected": selected,
                            "selection_probability": selection.probabilities[index],
                            "selection_effective_score": selection.effective_scores[index],
                            "redundancy_penalty": selection.redundancy_penalties[index],
                            "stop_reasons": list(decision.reasons) if selected else [],
                        }
                    )
                    trace.append(
                        {
                            "region_id": child.region_id,
                            "parent_id": child.parent_id,
                            "depth": child.depth,
                            "core_xyxy": list(child.core_xyxy),
                            "view_xyxy": list(child.view_xyxy),
                            "fused_score": child.score,
                            "score_components": child.score_components,
                            "active_scorers": list(active_scorers),
                            "selection_probability": selection.probabilities[index],
                            "selection_effective_score": selection.effective_scores[index],
                            "redundancy_penalty": selection.redundancy_penalties[index],
                            "selected": selected,
                            "stop_reasons": list(decision.reasons) if selected else [],
                            "sibling_cumulative_mass": selection.cumulative_probability,
                        }
                    )
                    if not selected:
                        continue
                    if decision.stop:
                        leaves.append(child)
                    else:
                        next_frontier.append(child)
            frontier = next_frontier
        if not leaves:
            leaves = [root]
        search_latency = (time.perf_counter() - search_started) * 1000.0

        fusion_started = time.perf_counter()
        fused = self.fusion.fuse(leaves, image_width, image_height)
        fusion_latency = (time.perf_counter() - fusion_started) * 1000.0
        total_latency = (time.perf_counter() - started) * 1000.0
        provenance: dict[str, Any] = {
            "locator": self.provider_name,
            "coordinate_mode": "absolute_original_pixel_xyxy",
            "detector": (
                {
                    "provider": proposals.provider,
                    "model_id": proposals.model_id,
                    "metadata": dict(proposals.metadata),
                }
                if proposals is not None
                else None
            ),
            "retriever": (
                {
                    "provider": self.retriever_provider.provider_name,
                    "model_id": str(getattr(self.retriever_provider, "model_id", "unknown")),
                }
                if self.retriever_provider is not None and plan.use_retrieval
                else None
            ),
        }
        return LocatorResult(
            regions_xyxy=tuple(region.view_xyxy for region in fused),
            scores=tuple(region.score for region in fused),
            task_spec=task,
            search_plan=plan,
            search_trace=tuple(trace),
            inspected_area_ratio=cumulative_inspected_area / image_area,
            depth_reached=max((item["depth"] for item in trace), default=0),
            latency_ms={
                "parser": parser_latency,
                "detector": detector_latency,
                "retrieval": retrieval_latency,
                "search": search_latency,
                "fusion": fusion_latency,
                "total": total_latency,
            },
            provider_provenance=provenance,
            warnings=tuple(dict.fromkeys(warnings)),
            region_details=tuple(region.to_dict() for region in fused),
        )

    def close(self) -> None:
        if self.detector_provider is not None:
            self.detector_provider.close()
        if self.retriever_provider is not None:
            self.retriever_provider.close()
