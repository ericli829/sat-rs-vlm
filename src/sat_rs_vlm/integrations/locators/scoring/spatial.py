"""Deterministic spatial priors with no model dependency."""

from __future__ import annotations

from collections.abc import Sequence

from sat_rs_vlm.semantics import TaskSpec

from ..geometry import bbox_coverage, spatial_prior
from ..types import SearchRegion
from .protocol import ScoreBatch


class SpatialRegionScorer:
    scorer_name = "spatial"

    def score(
        self,
        task: TaskSpec,
        regions: Sequence[SearchRegion],
        image_width: int,
        image_height: int,
    ) -> ScoreBatch:
        if task.relations and task.given_bbox is None:
            return ScoreBatch.unavailable(
                self.scorer_name,
                len(regions),
                "object_relation_requires_anchor_bbox",
            )
        if task.given_bbox is None and task.spatial_scope == "global":
            return ScoreBatch.unavailable(
                self.scorer_name,
                len(regions),
                "no_deterministic_spatial_constraint",
            )
        if task.given_bbox is not None:
            scores = tuple(
                bbox_coverage(task.given_bbox, region.core_xyxy) for region in regions
            )
            mode = "given_bbox_coverage"
        else:
            scores = tuple(
                spatial_prior(
                    region.core_xyxy,
                    image_width,
                    image_height,
                    task.spatial_scope,
                )
                for region in regions
            )
            mode = task.spatial_scope
        return ScoreBatch(
            name=self.scorer_name,
            scores=scores,
            available=True,
            metadata=tuple(
                {"available": True, "mode": mode, "raw_score": score}
                for score in scores
            ),
        )
