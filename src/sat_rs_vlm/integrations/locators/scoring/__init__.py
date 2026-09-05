"""Replaceable scoring components for hierarchical localization."""

from .composite import (
    CompositeRegionScorer,
    CompositeScoreResult,
    depth_pool_normalize,
    sibling_normalize,
)
from .detector import DetectorRegionScorer
from .protocol import ScoreBatch
from .retrieval import RetrievalRegionScorer
from .spatial import SpatialRegionScorer

__all__ = [
    "CompositeRegionScorer",
    "CompositeScoreResult",
    "depth_pool_normalize",
    "DetectorRegionScorer",
    "RetrievalRegionScorer",
    "ScoreBatch",
    "SpatialRegionScorer",
    "sibling_normalize",
]
