from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .schema import PlannerTarget

_POSITION_ALIASES = {
    "top": "TOP",
    "top edge": "TOP",
    "bottom": "BOTTOM",
    "bottom edge": "BOTTOM",
    "left": "LEFT",
    "left edge": "LEFT",
    "right": "RIGHT",
    "right edge": "RIGHT",
    "center": "CENTER",
    "centre": "CENTER",
    "middle": "CENTER",
    "top left": "TOP_LEFT",
    "top left corner": "TOP_LEFT",
    "top right": "TOP_RIGHT",
    "top right corner": "TOP_RIGHT",
    "bottom left": "BOTTOM_LEFT",
    "bottom left corner": "BOTTOM_LEFT",
    "bottom right": "BOTTOM_RIGHT",
    "bottom right corner": "BOTTOM_RIGHT",
    "top center": "TOP_CENTER",
    "top centre": "TOP_CENTER",
    "bottom center": "BOTTOM_CENTER",
    "bottom centre": "BOTTOM_CENTER",
    "center left": "CENTER_LEFT",
    "centre left": "CENTER_LEFT",
    "middle left": "CENTER_LEFT",
    "center right": "CENTER_RIGHT",
    "centre right": "CENTER_RIGHT",
    "middle right": "CENTER_RIGHT",
}
_RELATION_ALIASES = {
    "left of": "LEFT_OF",
    "right of": "RIGHT_OF",
    "above": "ABOVE",
    "below": "BELOW",
    "upper left of": "UPPER_LEFT_OF",
    "upper right of": "UPPER_RIGHT_OF",
    "lower left of": "LOWER_LEFT_OF",
    "lower right of": "LOWER_RIGHT_OF",
    "near": "NEAR",
    "next to": "NEXT_TO",
    "inside": "INSIDE",
    "outside": "OUTSIDE",
    "between": "BETWEEN",
    "around": "AROUND",
    "in front of": "IN_FRONT_OF",
    "behind": "BEHIND",
}
_ORDER_ALIASES = {
    "ascending": "ASCENDING",
    "descending": "DESCENDING",
    "top to bottom": "TOP_TO_BOTTOM",
    "bottom to top": "BOTTOM_TO_TOP",
    "left to right": "LEFT_TO_RIGHT",
    "right to left": "RIGHT_TO_LEFT",
}
_DIRECTION_ALIASES = {
    "leftmost": "LEFTMOST",
    "rightmost": "RIGHTMOST",
    "topmost": "TOPMOST",
    "bottommost": "BOTTOMMOST",
}
_SUBREGION_ALIASES = {
    "left side": "LEFT_SIDE",
    "right side": "RIGHT_SIDE",
    "above": "ABOVE",
    "below": "BELOW",
    "inside": "INSIDE",
    "outside": "OUTSIDE",
    "both sides": "BOTH_SIDES",
    "around": "AROUND",
}
_MODE_ALIASES = {
    "relation": "RELATION",
    "rank": "RANK",
    "ordinal": "ORDINAL",
    "extreme": "EXTREME",
    "subregion": "SUBREGION",
}
_NORMALIZERS = {
    "position": _POSITION_ALIASES,
    "relation": _RELATION_ALIASES,
    "order": _ORDER_ALIASES,
    "direction": _DIRECTION_ALIASES,
    "subregion": _SUBREGION_ALIASES,
    "mode": _MODE_ALIASES,
}
_SIMPLE_CATEGORY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _alias_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def normalize_candidate_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply only deterministic, semantics-preserving v1-to-v1.1 rewrites."""
    normalized = deepcopy(dict(payload))
    changes: list[dict[str, Any]] = []
    raw_final = normalized.get("final")
    if isinstance(raw_final, dict):
        if "source" in raw_final and "sources" not in raw_final:
            previous = raw_final.pop("source")
            replacement = [previous]
            raw_final["sources"] = replacement
            changes.append(
                {
                    "path": "final.source",
                    "from": previous,
                    "to": replacement,
                    "rule": "migrate_legacy_final_source",
                }
            )
        elif isinstance(raw_final.get("sources"), str):
            previous = raw_final["sources"]
            replacement = [previous]
            raw_final["sources"] = replacement
            changes.append(
                {
                    "path": "final.sources",
                    "from": previous,
                    "to": replacement,
                    "rule": "wrap_single_final_source",
                }
            )
    nodes = normalized.get("nodes")
    if not isinstance(nodes, list):
        return normalized, changes
    for index, raw_node in enumerate(nodes):
        if not isinstance(raw_node, dict):
            continue
        node_path = f"nodes[{index}]"
        if "output" in raw_node:
            previous = raw_node.pop("output")
            changes.append(
                {
                    "path": f"{node_path}.output",
                    "from": previous,
                    "to": None,
                    "rule": "remove_legacy_graphnode_output",
                }
            )
        params = raw_node.get("params")
        if not isinstance(params, dict):
            continue
        target = params.get("target")
        if isinstance(target, str) and _SIMPLE_CATEGORY.fullmatch(target.strip()):
            wrapped = {"category": target.strip(), "attributes": {}}
            params["target"] = wrapped
            changes.append(
                {
                    "path": f"{node_path}.params.target",
                    "from": target,
                    "to": wrapped,
                    "rule": "wrap_unambiguous_target_spec",
                }
            )
        for field, aliases in _NORMALIZERS.items():
            value = params.get(field)
            if not isinstance(value, str):
                continue
            replacement = aliases.get(_alias_key(value))
            if replacement is None or replacement == value:
                continue
            params[field] = replacement
            changes.append(
                {
                    "path": f"{node_path}.params.{field}",
                    "from": value,
                    "to": replacement,
                    "rule": "canonical_enum_alias",
                }
            )
    return normalized, changes


def _refs(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith("$n") else set()
    if isinstance(value, list):
        return set().union(*(_refs(item) for item in value), set())
    if isinstance(value, Mapping):
        return set().union(*(_refs(item) for item in value.values()), set())
    return set()


def _replace_refs(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_refs(item, mapping) for item in value]
    if isinstance(value, Mapping):
        return {key: _replace_refs(child, mapping) for key, child in value.items()}
    return value


def _without_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_none(child) for key, child in value.items() if child is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def canonicalize_target(target: PlannerTarget | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(target, PlannerTarget):
        parsed = target
    else:
        normalized, _ = normalize_candidate_payload(target)
        parsed = PlannerTarget.model_validate(normalized)
    nodes = [node.model_dump(mode="json", exclude_none=False) for node in parsed.nodes]
    remaining = list(nodes)
    ordered: list[dict[str, Any]] = []
    emitted: set[str] = set()
    while remaining:
        ready = [
            node
            for node in remaining
            if {ref[1:] for ref in _refs(node["inputs"])}.issubset(emitted)
        ]
        if not ready:
            raise ValueError("cannot canonicalize a cyclic or unresolved graph")
        node = ready[0]
        ordered.append(node)
        emitted.add(node["id"])
        remaining.remove(node)
    mapping = {f"${node['id']}": f"$n{index}" for index, node in enumerate(ordered, 1)}
    for index, node in enumerate(ordered, 1):
        node["id"] = f"n{index}"
        node["inputs"] = _replace_refs(node["inputs"], mapping)
    payload = {
        "intent": parsed.intent.value if parsed.intent is not None else None,
        "nodes": ordered,
        "final": {
            "sources": [mapping.get(ref, ref) for ref in parsed.final.sources],
            "question": parsed.final.question,
            "answer_type": parsed.final.answer_type.value,
        },
    }
    return _without_none(payload)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
