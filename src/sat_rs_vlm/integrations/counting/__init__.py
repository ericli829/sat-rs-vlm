"""COUNT capability adapter: 04_counting_system_plan -> TaskGraph CountingProvider."""

from .adapter import CountingSystemProvider
from .bootstrap import counting_system_src, ensure_counting_system_importable
from .bridge import to_counting_scope, to_counting_target, to_taskgraph_entity_set
from .detector_bridge import CountingProposalDetectorBridge

__all__ = [
    "CountingProposalDetectorBridge",
    "CountingSystemProvider",
    "counting_system_src",
    "ensure_counting_system_importable",
    "to_counting_scope",
    "to_counting_target",
    "to_taskgraph_entity_set",
]
