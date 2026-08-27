"""Compatibility exports for the common production semantic layer."""

from sat_rs_vlm.semantics import (
    SemanticFacts,
    TermMention,
    extract_semantic_facts,
    load_ontology,
)

__all__ = [
    "SemanticFacts",
    "TermMention",
    "extract_semantic_facts",
    "load_ontology",
]
