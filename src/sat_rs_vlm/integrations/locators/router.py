"""Task-driven routing that is independent of dataset names and model classes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sat_rs_vlm.semantics import TaskSpec

from .types import SearchPlan


class TaskRouter:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def route(self, task: TaskSpec) -> SearchPlan:
        warnings: list[str] = []
        targets = tuple(task.targets)
        has_target = bool(targets)
        has_spatial = task.spatial_scope != "global" or task.given_bbox is not None
        if task.given_bbox is not None:
            return SearchPlan(
                use_detector=False,
                use_retrieval=False,
                use_spatial=True,
                bypass_locator=True,
                desired_multi_region=False,
                target_phrases=targets,
                route="given_bbox_direct",
            )
        if task.operation == "global_scene":
            return SearchPlan(
                use_detector=False,
                use_retrieval=False,
                use_spatial=False,
                bypass_locator=True,
                desired_multi_region=False,
                route="global_view",
            )
        if task.operation in {"count", "existence"}:
            if not has_target:
                warnings.append("detector_disabled_without_target")
            return SearchPlan(
                use_detector=has_target,
                use_retrieval=bool(self.config.get("retrieval_for_counting", True)),
                use_spatial=has_spatial,
                bypass_locator=False,
                desired_multi_region=task.operation == "count",
                target_phrases=targets,
                route="detector_first",
                warnings=tuple(warnings),
            )
        if task.operation in {"attribute", "category"}:
            if not has_target:
                warnings.append("detector_disabled_without_target")
            return SearchPlan(
                use_detector=has_target,
                use_retrieval=True,
                use_spatial=has_spatial,
                bypass_locator=False,
                desired_multi_region=False,
                target_phrases=targets,
                route="object_attribute",
                warnings=tuple(warnings),
            )
        if task.operation in {"position", "grounding"}:
            return SearchPlan(
                use_detector=has_target,
                use_retrieval=True,
                use_spatial=has_spatial,
                bypass_locator=False,
                desired_multi_region=False,
                target_phrases=targets,
                route="grounding",
            )
        if task.operation == "relation":
            return SearchPlan(
                use_detector=has_target,
                use_retrieval=True,
                use_spatial=has_spatial,
                bypass_locator=False,
                desired_multi_region=True,
                target_phrases=targets,
                route="relation_multi_source",
            )
        if task.operation == "open_reasoning":
            return SearchPlan(
                use_detector=False,
                use_retrieval=True,
                use_spatial=has_spatial,
                bypass_locator=False,
                desired_multi_region=True,
                route="retrieval_first",
            )
        return SearchPlan(
            use_detector=False,
            use_retrieval=True,
            use_spatial=has_spatial,
            bypass_locator=False,
            desired_multi_region=True,
            route="safe_retrieval_default",
            warnings=("unknown_task_safe_route",),
        )
