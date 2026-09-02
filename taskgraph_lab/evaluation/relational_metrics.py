"""Structure-focused metrics for relation-heavy TaskGraph predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_RELATIONAL_INTENTS = {"RELATIONAL_COUNT", "OBJECT_RELATION"}


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.startswith("$n") else []
    if isinstance(value, list):
        return [ref for item in value for ref in _references(item)]
    if isinstance(value, Mapping):
        return [ref for item in value.values() for ref in _references(item)]
    return []


def _relation_nodes(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        node
        for node in graph.get("nodes") or []
        if node.get("op") == "RELATION"
        or (node.get("op") == "SELECT" and (node.get("params") or {}).get("mode") == "RELATION")
    ]


def _relation_signature(node: Mapping[str, Any]) -> tuple[Any, ...]:
    params = node.get("params") or {}
    inputs = node.get("inputs") or {}
    if node.get("op") == "RELATION":
        return ("RELATION", inputs.get("subject"), inputs.get("reference"), None)
    return (
        "SELECT_REL",
        inputs.get("candidates"),
        inputs.get("reference"),
        params.get("relation"),
    )


def _dependencies(graph: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        str(node["id"]): {ref[1:] for ref in _references(node.get("inputs") or {})}
        for node in graph.get("nodes") or []
    }


def _ancestors(node_id: str, dependencies: Mapping[str, set[str]]) -> set[str]:
    found: set[str] = set()
    pending = list(dependencies.get(node_id, set()))
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        pending.extend(dependencies.get(current, set()))
    return found


def count_consumes_filtered_source(graph: Mapping[str, Any]) -> bool | None:
    nodes = {str(node["id"]): node for node in graph.get("nodes") or []}
    select_ids = {
        node_id
        for node_id, node in nodes.items()
        if node.get("op") == "SELECT" and (node.get("params") or {}).get("mode") == "RELATION"
    }
    counts = [node for node in nodes.values() if node.get("op") == "COUNT"]
    if not select_ids or not counts:
        return None
    dependencies = _dependencies(graph)
    return any(bool(_ancestors(str(node["id"]), dependencies) & select_ids) for node in counts)


def _count_scope(graph: Mapping[str, Any]) -> list[tuple[str, Any]]:
    result = []
    for node in graph.get("nodes") or []:
        if node.get("op") != "COUNT":
            continue
        inputs = node.get("inputs") or {}
        role = "entities" if "entities" in inputs else "image" if "image" in inputs else "unknown"
        result.append((role, (node.get("params") or {}).get("entire")))
    return result


def relational_metrics(
    expected: Mapping[str, Any],
    predicted: Mapping[str, Any] | None,
    *,
    validation_error_codes: Sequence[str] = (),
) -> dict[str, Any]:
    applicable = str(expected.get("intent")) in _RELATIONAL_INTENTS
    base: dict[str, Any] = {
        "relational_metric_applicable": applicable,
        "relation_direction_accuracy": None,
        "reference_attachment_accuracy": None,
        "select_rel_vs_relation_accuracy": None,
        "count_filtered_source_accuracy": None,
        "count_scope_entire_accuracy": None,
        "dead_node": "dead_node" in validation_error_codes,
        "first_broken_relation_chain_depth": None,
    }
    if not applicable or predicted is None:
        return base
    expected_nodes = _relation_nodes(expected)
    predicted_nodes = _relation_nodes(predicted)
    expected_signatures = [_relation_signature(node) for node in expected_nodes]
    predicted_signatures = [_relation_signature(node) for node in predicted_nodes]
    expected_directions = [signature[3] for signature in expected_signatures]
    predicted_directions = [signature[3] for signature in predicted_signatures]
    base["relation_direction_accuracy"] = expected_directions == predicted_directions
    base["reference_attachment_accuracy"] = [
        signature[1:3] for signature in expected_signatures
    ] == [signature[1:3] for signature in predicted_signatures]
    base["select_rel_vs_relation_accuracy"] = [
        signature[0] for signature in expected_signatures
    ] == [signature[0] for signature in predicted_signatures]
    if str(expected.get("intent")) == "RELATIONAL_COUNT":
        base["count_filtered_source_accuracy"] = (
            count_consumes_filtered_source(expected) == count_consumes_filtered_source(predicted)
        )
        base["count_scope_entire_accuracy"] = _count_scope(expected) == _count_scope(predicted)
    shared = min(len(expected_signatures), len(predicted_signatures))
    broken = next(
        (
            index + 1
            for index in range(shared)
            if expected_signatures[index] != predicted_signatures[index]
        ),
        None,
    )
    if broken is None and len(expected_signatures) != len(predicted_signatures):
        broken = shared + 1
    base["first_broken_relation_chain_depth"] = broken
    return base

