"""COUNT capability adapter: 04_counting_system_plan -> TaskGraph DetectionProvider."""

from .adapter import CountingSystemDetectionAdapter
from .bootstrap import counting_system_src, ensure_counting_system_importable
from .bridge import to_counting_scope, to_counting_target, to_taskgraph_entity_set

__all__ = [
    "CountingSystemDetectionAdapter",
    "counting_system_src",
    "ensure_counting_system_importable",
    "to_counting_scope",
    "to_counting_target",
    "to_taskgraph_entity_set",
]
