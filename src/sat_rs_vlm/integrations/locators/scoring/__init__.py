"""Replaceable scoring components for hierarchical localization."""

from .composite import CompositeRegionScorer, CompositeScoreResult, sibling_normalize
from .detector import DetectorRegionScorer
from .protocol import ScoreBatch
from .retrieval import RetrievalRegionScorer
from .spatial import SpatialRegionScorer

__all__ = [
    "CompositeRegionScorer",
    "CompositeScoreResult",
    "DetectorRegionScorer",
    "RetrievalRegionScorer",
    "ScoreBatch",
    "SpatialRegionScorer",
    "sibling_normalize",
]
