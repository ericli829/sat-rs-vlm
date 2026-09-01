"""Frozen TaskGraph operator input-role and runtime-type contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputRoleContract:
    runtime_types: frozenset[str]
    allow_list: bool = False


@dataclass(frozen=True)
class OperatorInputContract:
    roles: dict[str, InputRoleContract]
    required: frozenset[str] = field(default_factory=frozenset)
    exactly_one: tuple[frozenset[str], ...] = ()
    require_any: frozenset[str] = field(default_factory=frozenset)

    def validate_keys(self, inputs: dict[str, Any], *, operator: str) -> None:
        keys = set(inputs)
        unexpected = keys - set(self.roles)
        missing = set(self.required) - keys
        if unexpected:
            raise ValueError(f"{operator} has unexpected input role(s): {sorted(unexpected)}")
        if missing:
            raise ValueError(f"{operator} is missing required input role(s): {sorted(missing)}")
        for group in self.exactly_one:
            present = keys.intersection(group)
            if len(present) != 1:
                raise ValueError(
                    f"{operator} requires exactly one of {sorted(group)}, got {sorted(present)}"
                )
        if self.require_any and not keys.intersection(self.require_any):
            raise ValueError(f"{operator} requires at least one of {sorted(self.require_any)}")
        for role, value in inputs.items():
            is_list = isinstance(value, list)
            if is_list and not self.roles[role].allow_list:
                raise ValueError(f"{operator}.{role} does not allow reference lists")
            if is_list and not value:
                raise ValueError(f"{operator}.{role} reference list must not be empty")


def _role(*runtime_types: str, allow_list: bool = False) -> InputRoleContract:
    return InputRoleContract(frozenset(runtime_types), allow_list)


VISUAL_SCOPE = ("ImageRef", "Region")
ENTITY_OR_REGION = ("Entity", "EntitySet", "Region")
ANY_RUNTIME_TYPES = frozenset(
    {
        "ImageRef",
        "Region",
        "RegionSet",
        "Entity",
        "EntitySet",
        "SelectResult",
        "ScalarInt",
        "ScalarFloat",
        "Boolean",
        "Label",
        "LabelSet",
        "RouteContext",
        "Evidence",
        "EvidenceSet",
        "Answer",
    }
)


OPERATOR_INPUT_CONTRACTS: dict[str, OperatorInputContract] = {
    "REGION": OperatorInputContract({"image": _role(*VISUAL_SCOPE)}, frozenset({"image"})),
    "REGION_FROM_BBOX": OperatorInputContract({"image": _role("ImageRef")}, frozenset({"image"})),
    "FIND_MARKER": OperatorInputContract({"image": _role(*VISUAL_SCOPE)}, frozenset({"image"})),
    "LOCATE": OperatorInputContract({"image": _role(*VISUAL_SCOPE)}, frozenset({"image"})),
    "SELECT": OperatorInputContract(
        {
            "candidates": _role("EntitySet", "Region", "RegionSet", "SelectResult"),
            "reference": _role(*ENTITY_OR_REGION, "SelectResult"),
            # Current global search scope.  Required by precise SUBREGION
            # composition, optional for backwards compatible graphs.
            "scope": _role(*VISUAL_SCOPE),
        },
        frozenset({"candidates"}),
    ),
    "GROUP": OperatorInputContract({"entities": _role("EntitySet")}, frozenset({"entities"})),
    "COUNT": OperatorInputContract(
        {"image": _role(*VISUAL_SCOPE), "entities": _role("EntitySet", "SelectResult")},
        exactly_one=(frozenset({"image", "entities"}),),
    ),
    "ATTRIBUTE": OperatorInputContract({"entity": _role(*ENTITY_OR_REGION)}, frozenset({"entity"})),
    "CLASSIFY": OperatorInputContract(
        {"source": _role("ImageRef", "Region", "Entity")}, frozenset({"source"})
    ),
    "MULTILABEL_CLASSIFY": OperatorInputContract(
        {"source": _role("ImageRef", "Region")}, frozenset({"source"})
    ),
    "MOTION": OperatorInputContract(
        {"source": _role("Region", "Entity", "EntitySet")}, frozenset({"source"})
    ),
    "RELATION": OperatorInputContract(
        {
            "subject": _role(*ENTITY_OR_REGION),
            "reference": _role(*ENTITY_OR_REGION),
        },
        frozenset({"subject", "reference"}),
    ),
    "ABS_DIFF": OperatorInputContract(
        {"a": _role("ScalarInt"), "b": _role("ScalarInt")}, frozenset({"a", "b"})
    ),
    "VLM_REASON": OperatorInputContract(
        {
            "image": _role(*VISUAL_SCOPE),
            "evidence": InputRoleContract(ANY_RUNTIME_TYPES, allow_list=True),
        },
        require_any=frozenset({"image", "evidence"}),
    ),
    "BUILD_ROUTE_CONTEXT": OperatorInputContract(
        {
            "image": _role(*VISUAL_SCOPE),
            "start": _role(*ENTITY_OR_REGION),
            "goal": _role(*ENTITY_OR_REGION),
        },
        frozenset({"image", "start", "goal"}),
    ),
    "ROUTE_REASON": OperatorInputContract(
        {"context": _role("RouteContext")}, frozenset({"context"})
    ),
    # Loader/runtime compatibility only. New planner exports should use final.sources.
    "MATCH_CHOICE": OperatorInputContract(
        {
            "value": _role(
                "ScalarInt",
                "ScalarFloat",
                "Boolean",
                "Label",
                "LabelSet",
                "Answer",
            )
        },
        frozenset({"value"}),
    ),
}

DEPRECATED_OPERATORS = frozenset({"MATCH_CHOICE"})


def validate_runtime_inputs(operator: str, inputs: dict[str, Any]) -> None:
    contract = OPERATOR_INPUT_CONTRACTS[operator]
    contract.validate_keys(inputs, operator=operator)
    for role, value in inputs.items():
        values = value if isinstance(value, list) else [value]
        allowed = contract.roles[role].runtime_types
        actual = [type(item).__name__ for item in values]
        invalid = [name for name in actual if name not in allowed]
        if invalid:
            raise TypeError(f"{operator}.{role} expected {sorted(allowed)}, got {actual}")
