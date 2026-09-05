"""Common remote-sensing semantics for runtime routing and evaluation."""

from .mentions import extract_semantic_facts
from .ontology import load_ontology
from .question_parser import OptionalFallbackQueryParser, QueryParser, RuleBasedQueryParser
from .types import RelationSpec, SemanticFacts, TaskSpec, TermMention

__all__ = [
    "OptionalFallbackQueryParser",
    "QueryParser",
    "RelationSpec",
    "RuleBasedQueryParser",
    "SemanticFacts",
    "TaskSpec",
    "TermMention",
    "extract_semantic_facts",
    "load_ontology",
]
