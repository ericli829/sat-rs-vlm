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
from .geometry import (
    bbox_area,
    clamp_bbox,
    expand_with_halo,
    rectangle_union_area,
    subdivide_core,
)
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
    min_beam: int = 1
    score_threshold: float | None = None
    spatial_prefilter: bool = False
    spatial_first_depth_only: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> HierarchicalSearchConfig:
        values = dict(config or {})
        result = cls(
            grid_size=int(values.get("grid_size", cls.grid_size)),
            halo_ratio=float(values.get("halo_ratio", cls.halo_ratio)),
            temperature=float(values.get("temperature", cls.temperature)),
            cumulative_mass=float(values.get("cumulative_mass", cls.cumulative_mass)),
            max_beam=int(values.get("max_beam", cls.max_beam)),
            min_beam=int(values.get("min_beam", cls.min_beam)),
            score_threshold=(
                float(values["score_threshold"])
                if values.get("score_threshold") is not None
                else None
            ),
            spatial_prefilter=bool(values.get("spatial_prefilter", cls.spatial_prefilter)),
            spatial_first_depth_only=bool(
                values.get("spatial_first_depth_only", cls.spatial_first_depth_only)
            ),
        )
        if result.grid_size < 2 or result.max_beam < 1 or result.min_beam < 1:
            raise LocatorError("search grid_size/max_beam/min_beam values are invalid")
        if result.min_beam > result.max_beam:
            raise LocatorError("search min_beam must not exceed max_beam")
        if result.halo_ratio < 0.0 or result.temperature <= 0.0:
            raise LocatorError("search halo_ratio/temperature values are invalid")
        if not 0.0 < result.cumulative_mass <= 1.0:
            raise LocatorError("search cumulative_mass must be in (0, 1]")
        if result.score_threshold is not None and not 0.0 <= result.score_threshold <= 1.0:
            raise LocatorError("search score_threshold must be between 0 and 1")
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

    def _initial_search_box(self, scope: str, image_width: int, image_height: int) -> BBox:
        """Use a soft directional window before the first grid expansion."""

        full = (0.0, 0.0, float(image_width), float(image_height))
        if not self.search_config.spatial_prefilter or scope == "global":
            return full
        x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
        if scope in {"left", "west", "upper_left", "lower_left", "center_left"}:
            x2 = 2.0 / 3.0
        elif scope in {"right", "east", "upper_right", "lower_right", "center_right"}:
            x1 = 1.0 / 3.0
        if scope in {"upper", "north", "upper_left", "upper_right"}:
            y2 = 2.0 / 3.0
        elif scope in {"lower", "south", "lower_left", "lower_right"}:
            y1 = 1.0 / 3.0
        if scope == "center":
            x1, y1, x2, y2 = 0.25, 0.25, 0.75, 0.75
        core = (x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height)
        return expand_with_halo(
            core,
            self.search_config.halo_ratio,
            image_width,
            image_height,
        )

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
                    "proposal_count": len(result.boxes_xyxy),
                    "boxes_xyxy": [list(box) for box in result.boxes_xyxy],
                    "scores": list(result.scores),
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
                metadata={
                    "queries": sources,
                    "coordinate_mode": "absolute_pixel_xyxy",
                    "proposal_count": len(boxes),
                    "boxes_xyxy": boxes,
                    "scores": scores,
                },
            ),
            latency,
        )

    @staticmethod
    def _retrieval_query(task: TaskSpec, depth: int) -> str:
        """Use direction only for the coarse pass, then retrieve by target class."""

        if depth <= 1 or task.spatial_scope == "global":
            return task.raw_question
        if task.targets:
            return ", ".join(target.replace("_", " ") for target in task.targets)
        return task.raw_question

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
        parent_scores: Sequence[float],
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
                    self._retrieval_query(task, children[0].depth if children else 1),
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
        spatial_allowed = not self.search_config.spatial_first_depth_only or (
            bool(children) and all(child.depth == 1 for child in children)
        )
        if plan.use_spatial and self.spatial_enabled and spatial_allowed:
            batches.append(
                self.spatial_scorer.score(task, children, image_width, image_height)
            )
        else:
            batches.append(
                ScoreBatch.unavailable(
                    "spatial",
                    len(children),
                    (
                        "spatial_only_first_depth"
                        if plan.use_spatial
                        and self.spatial_enabled
                        and self.search_config.spatial_first_depth_only
                        else "disabled_by_plan_or_config"
                    ),
                )
            )
        composite = self.composite_scorer.score(
            children,
            batches,
            parent_scores=parent_scores,
        )
        return (
            composite.scores,
            composite.components,
            retrieval_latency,
            composite.active_scorers,
        )

    @staticmethod
    def _retrieval_query(task: TaskSpec, depth: int) -> str:
        """Use direction only for the coarse pass, then retrieve by target class."""

        if depth <= 1 or task.spatial_scope == "global":
            return task.raw_question
        if task.targets:
            return ", ".join(target.replace("_", " ") for target in task.targets)
        return task.raw_question

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
            processed_area_ratio=bbox_area(box) / (image_width * image_height),
            selected_union_area_ratio=bbox_area(box) / (image_width * image_height),
            processed_union_area_ratio=bbox_area(box) / (image_width * image_height),
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
        root_box = self._initial_search_box(
            task.spatial_scope,
            image_width,
            image_height,
        )
        root = SearchRegion(
            region_id="root",
            parent_id=None,
            depth=0,
            core_xyxy=root_box,
            view_xyxy=root_box,
            score=0.0,
            metadata={
                "coordinate_mode": "absolute_original_pixel_xyxy",
                "spatial_prefilter": self.search_config.spatial_prefilter,
                "spatial_scope": task.spatial_scope,
            },
        )
        frontier = [root]
        leaves: list[SearchRegion] = []
        trace: list[dict[str, Any]] = []
        evaluated_regions = 0
        cumulative_inspected_area = 0.0
        processed_views: list[BBox] = []
        image_area = float(image_width * image_height)
        retrieval_latency = 0.0
        search_started = time.perf_counter()

        while frontier:
            child_count = self.search_config.grid_size**2
            depth_candidate_count = len(frontier) * child_count
            remaining_budget = self.stop_config.max_regions - evaluated_regions
            if remaining_budget < depth_candidate_count:
                for parent in frontier:
                    parent.metadata["stop_reasons"] = [
                        "max_regions_before_depth_expansion"
                    ]
                leaves.extend(frontier)
                break

            candidate_children: list[SearchRegion] = []
            parent_scores: list[float] = []
            for parent in frontier:
                parent_boxes = subdivide_core(
                    parent.core_xyxy,
                    self.search_config.grid_size,
                )
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
                        metadata={
                            "coordinate_mode": "absolute_original_pixel_xyxy",
                            "parent_score": parent.score,
                        },
                    )
                    for index, box in enumerate(parent_boxes)
                ]
                parent.children.extend(child.region_id for child in children)
                candidate_children.extend(children)
                parent_scores.extend([parent.score] * len(children))

            evaluated_regions += len(candidate_children)
            processed_views.extend(child.view_xyxy for child in candidate_children)
            cumulative_inspected_area += sum(
                bbox_area(child.view_xyxy) for child in candidate_children
            )
            processed_ratio = cumulative_inspected_area / image_area
            scores, components, batch_latency, active_scorers = self._score_children(
                image_path=resolved_image,
                task=task,
                plan=plan,
                children=candidate_children,
                proposals=proposals,
                image_width=image_width,
                image_height=image_height,
                parent_scores=parent_scores,
                warnings=warnings,
            )
            retrieval_latency += batch_latency
            for child, score, detail in zip(
                candidate_children,
                scores,
                components,
                strict=True,
            ):
                child.score = score
                child.score_components = detail

            selection = adaptive_beam_select(
                candidate_children,
                scores,
                temperature=self.search_config.temperature,
                cumulative_mass=self.search_config.cumulative_mass,
                max_beam=self.search_config.max_beam,
                redundancy_weight=self.composite_scorer.weights["redundancy"],
                score_threshold=self.search_config.score_threshold,
                min_beam=self.search_config.min_beam,
            )
            selected_set = set(selection.selected_indices)
            next_frontier: list[SearchRegion] = []
            selected_count = len(selected_set)
            for index, child in enumerate(candidate_children):
                decision = evaluate_stop(
                    child,
                    self.stop_config,
                    evaluated_regions=evaluated_regions,
                    processed_area_ratio=processed_ratio,
                )
                selected = index in selected_set
                child.metadata.update(
                    {
                        "selected": selected,
                        "selection_probability": selection.probabilities[index],
                        "beam_standardized_logit": selection.standardized_logits[index],
                        "selection_effective_logit": selection.effective_scores[index],
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
                        "beam_standardized_logit": selection.standardized_logits[index],
                        "score_components": child.score_components,
                        "active_scorers": list(active_scorers),
                        "selection_probability": selection.probabilities[index],
                        "selection_effective_logit": selection.effective_scores[index],
                        "redundancy_penalty": selection.redundancy_penalties[index],
                        "selected": selected,
                        "stop_reasons": list(decision.reasons) if selected else [],
                        "depth_cumulative_mass": selection.cumulative_probability,
                        "depth_candidate_count": len(candidate_children),
                        "depth_selected_count": selected_count,
                        "beam_entropy": selection.entropy,
                        "processed_area_ratio": processed_ratio,
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
        processed_area_ratio = cumulative_inspected_area / image_area
        processed_union_area_ratio = (
            rectangle_union_area(processed_views) / image_area if processed_views else 0.0
        )
        selected_union_area_ratio = (
            rectangle_union_area(region.view_xyxy for region in fused) / image_area
            if fused
            else 0.0
        )
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
            processed_area_ratio=processed_area_ratio,
            selected_union_area_ratio=min(selected_union_area_ratio, 1.0),
            processed_union_area_ratio=min(processed_union_area_ratio, 1.0),
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
