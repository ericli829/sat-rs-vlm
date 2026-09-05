"""RetrieverProvider to locator scorer adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider

from ..types import SearchRegion
from .protocol import ScoreBatch


class RetrievalRegionScorer:
    scorer_name = "retrieval"

    def __init__(self, provider: RetrieverProvider | None) -> None:
        self.provider = provider

    def score(
        self,
        image_path: Path,
        query: str,
        regions: Sequence[SearchRegion],
    ) -> ScoreBatch:
        if self.provider is None:
            return ScoreBatch.unavailable(
                self.scorer_name,
                len(regions),
                "retriever_provider_unavailable",
            )
        result = self.provider.score_regions(
            image_path,
            query,
            [region.view_xyxy for region in regions],
        ).validate_length(len(regions))
        return ScoreBatch(
            name=self.scorer_name,
            scores=tuple(result.scores),
            available=True,
            metadata=tuple(
                {
                    "available": True,
                    "provider": result.provider,
                    "model_id": result.model_id,
                    "raw_score": score,
                    "latency_ms": result.latency_ms,
                    "provider_metadata": dict(result.metadata),
                }
                for score in result.scores
            ),
        )
