"""Query-aware UHR locator framework."""

from .answer import AnswerModel, AnswerResult, MultiROIRequest
from .protocol import LocatorProvider
from .registry import LOCATOR_NAMES, create_locator
from .types import LocatorError, LocatorResult, SearchPlan, SearchRegion

__all__ = [
    "AnswerModel",
    "AnswerResult",
    "LOCATOR_NAMES",
    "LocatorError",
    "LocatorProvider",
    "LocatorResult",
    "MultiROIRequest",
    "SearchPlan",
    "SearchRegion",
    "create_locator",
]
