"""Physical LOCATE requirements from directed TaskGraph input roles only."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .schema import GraphNode, OperatorName


class LocateRequirement(str, Enum):
    SET_REQUIRED = "SET_REQUIRED"
    SINGLE_TARGET = "SINGLE_TARGET"


@dataclass(frozen=True)
class LocateExecutionHint:
    requirement: LocateRequirement
    route_reason: str
    consumer_paths: tuple[str, ...]

    @property
    def backend(self) -> str:
        return "LAE_DINO" if self.requirement is LocateRequirement.SET_REQUIRED else "VLM_SINGLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement.value,
            "route_reason": self.route_reason,
            "consumer_paths": list(self.consumer_paths),
            "planned_backend": self.backend,
        }


def analyze_locate(node_id: str, nodes: Iterable[GraphNode]) -> LocateExecutionHint:
    """Reference edges do not transmit the candidate stream of a SELECT.

    A SUBREGION of a real Region is geometric slicing, not object enumeration.
    LOCATE itself yields EntitySet; SUBREGION directly on that uncertain set is
    conservative. Region-producing siblings never contaminate LOCATE routes.
    """
    graph = tuple(nodes)
    consumers: dict[str, list[tuple[GraphNode, str]]] = {}
    for consumer in graph:
        for role, raw in consumer.inputs.items():
            for ref in raw if isinstance(raw, list) else (raw,):
                consumers.setdefault(ref, []).append((consumer, role))
    set_paths: list[str] = []
    single_paths: list[str] = []
    ambiguous: list[str] = []
    pending = [(f"${node_id}", (node_id,))]
    visited = set()
    while pending:
        ref, path = pending.pop()
        if ref in visited:
            continue
        visited.add(ref)
        for consumer, role in consumers.get(ref, []):
            step = f"{consumer.id}:{consumer.op.value}.{role}"
            rendered = " -> ".join((*path, step))
            op = consumer.op
            if (
                (op is OperatorName.COUNT and role == "entities")
                or (op is OperatorName.GROUP and role == "entities")
                or (
                    op is OperatorName.SELECT
                    and role == "candidates"
                    and consumer.params["mode"] != "SUBREGION"
                )
            ):
                set_paths.append(rendered)
                if op in {OperatorName.SELECT, OperatorName.GROUP}:
                    pending.append((f"${consumer.id}", (*path, step)))
            elif op is OperatorName.SELECT and role == "candidates":
                ambiguous.append(rendered + " (EntitySet -> SUBREGION)")
                # A Region slice cannot propagate COUNT.image enumeration.
            elif role in {
                OperatorName.SELECT: {"reference"},
                OperatorName.ATTRIBUTE: {"entity"},
                OperatorName.CLASSIFY: {"source", "input"},
                OperatorName.MULTILABEL_CLASSIFY: {"source", "input"},
                OperatorName.MOTION: {"source", "input", "before", "after"},
                OperatorName.RELATION: {"subject", "reference"},
                OperatorName.BUILD_ROUTE_CONTEXT: {"start", "goal"},
            }.get(op, set()):
                single_paths.append(rendered)
            else:
                ambiguous.append(rendered)
                pending.append((f"${consumer.id}", (*path, step)))
    if set_paths:
        return LocateExecutionHint(
            LocateRequirement.SET_REQUIRED,
            "set_consumer",
            tuple(sorted(set_paths + ambiguous + single_paths)),
        )
    if ambiguous:
        return LocateExecutionHint(
            LocateRequirement.SET_REQUIRED,
            "ambiguous_set_requirement",
            tuple(sorted(ambiguous + single_paths)),
        )
    return LocateExecutionHint(
        LocateRequirement.SINGLE_TARGET,
        "single_consumer" if single_paths else "direct_localization",
        tuple(sorted(single_paths)),
    )
