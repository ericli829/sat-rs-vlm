"""Map global detector evidence to candidate regions using box coverage."""

from __future__ import annotations

from collections.abc import Sequence

from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.semantics import TaskSpec

from ..geometry import bbox_coverage
from ..types import SearchRegion
from .protocol import ScoreBatch


class DetectorRegionScorer:
    scorer_name = "detector"

    def score(
        self,
        task: TaskSpec,
        regions: Sequence[SearchRegion],
        proposals: ProposalResult | None,
    ) -> ScoreBatch:
        if not task.targets:
            return ScoreBatch.unavailable(
                self.scorer_name,
                len(regions),
                "task_has_no_detector_target",
            )
        if proposals is None:
            return ScoreBatch.unavailable(
                self.scorer_name,
                len(regions),
                "proposal_provider_unavailable",
            )
        scores: list[float] = []
        metadata: list[dict[str, object]] = []
        for region in regions:
            contributions = [
                float(confidence) * bbox_coverage(box, region.core_xyxy)
                for box, confidence in zip(
                    proposals.boxes_xyxy,
                    proposals.scores,
                    strict=True,
                )
            ]
            scores.append(sum(contributions))
            metadata.append(
                {
                    "available": True,
                    "proposal_count": len(contributions),
                    "nonzero_contributions": sum(value > 0.0 for value in contributions),
                    "contributions": contributions,
                    "provider": proposals.provider,
                    "model_id": proposals.model_id,
                }
            )
        return ScoreBatch(
            name=self.scorer_name,
            scores=tuple(scores),
            available=True,
            metadata=tuple(metadata),
        )
