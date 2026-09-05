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
    valid_role_sets: tuple[frozenset[str], ...] = ()

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
        if self.valid_role_sets and frozenset(keys) not in self.valid_role_sets:
            expected = [sorted(group) for group in self.valid_role_sets]
            raise ValueError(
                f"{operator} input roles must match one of {expected}, got {sorted(keys)}"
            )
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
        "ChoiceScoreResult",
    }
)


OPERATOR_INPUT_CONTRACTS: dict[str, OperatorInputContract] = {
    "REGION": OperatorInputContract({"image": _role(*VISUAL_SCOPE)}, frozenset({"image"})),
    "REGION_FROM_BBOX": OperatorInputContract(
        {"image": _role("ImageRef", "Region")}, frozenset({"image"})
    ),
    "FIND_MARKER": OperatorInputContract({"image": _role(*VISUAL_SCOPE)}, frozenset({"image"})),
    "LOCATE": OperatorInputContract({"image": _role(*VISUAL_SCOPE)}, frozenset({"image"})),
    "SELECT": OperatorInputContract(
        {
            "candidates": _role("EntitySet", "Region", "RegionSet", "SelectResult"),
            "reference": _role(*ENTITY_OR_REGION, "RegionSet", "SelectResult"),
            # Current global search scope.  Required by precise SUBREGION
            # composition, optional for backwards compatible graphs.
            "scope": _role(*VISUAL_SCOPE, "SelectResult"),
        },
        frozenset({"candidates"}),
    ),
    "GROUP": OperatorInputContract(
        {"entities": _role("EntitySet", "SelectResult")}, frozenset({"entities"})
    ),
    "COUNT": OperatorInputContract(
        {
            "image": _role(*VISUAL_SCOPE, "RegionSet"),
            "entities": _role("EntitySet", "SelectResult"),
        },
        exactly_one=(frozenset({"image", "entities"}),),
    ),
    "ATTRIBUTE": OperatorInputContract(
        {"entity": _role(*ENTITY_OR_REGION, "SelectResult")}, frozenset({"entity"})
    ),
    "CLASSIFY": OperatorInputContract(
        {"source": _role("ImageRef", "Region", "RegionSet", "Entity", "SelectResult")},
        frozenset({"source"}),
    ),
    "MULTILABEL_CLASSIFY": OperatorInputContract(
        {"source": _role("ImageRef", "Region", "RegionSet", "Entity", "SelectResult")},
        frozenset({"source"}),
    ),
    "MOTION": OperatorInputContract(
        {
            "source": _role("ImageRef", "Region", "Entity", "EntitySet", "SelectResult"),
            "before": _role("ImageRef", "Region", "Entity", "EntitySet", "SelectResult"),
            "after": _role("ImageRef", "Region", "Entity", "EntitySet", "SelectResult"),
        },
        valid_role_sets=(frozenset({"source"}), frozenset({"before", "after"})),
    ),
    "RELATION": OperatorInputContract(
        {
            # RegionSet (FIND_MARKER output) is valid relational evidence:
            # the semantic RELATION step renders every region crop and decides.
            "subject": _role(*ENTITY_OR_REGION, "RegionSet", "SelectResult"),
            "reference": _role(*ENTITY_OR_REGION, "RegionSet", "SelectResult"),
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
                "ChoiceScoreResult",
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


_NORMAL_SEMANTIC_OUTPUTS: dict[str, frozenset[str]] = {
    "ATTRIBUTE": frozenset({"Label"}),
    "CLASSIFY": frozenset({"Label"}),
    "MULTILABEL_CLASSIFY": frozenset({"LabelSet"}),
    "MOTION": frozenset({"Boolean"}),
    "RELATION": frozenset({"Label"}),
    "VLM_REASON": frozenset({"Answer"}),
    "MATCH_CHOICE": frozenset({"Answer"}),
    # ROUTE_REASON has always transported a precomputed score result.
    "ROUTE_REASON": frozenset({"ChoiceScoreResult"}),
}


def validate_runtime_output(
    operator: str,
    value: Any,
    *,
    final_choice_fusion: bool,
) -> None:
    """Validate execution-dependent VLM output types without changing the DAG schema."""

    normal = _NORMAL_SEMANTIC_OUTPUTS.get(operator)
    if normal is None:
        return
    allowed = frozenset({"ChoiceScoreResult"}) if final_choice_fusion else normal
    actual = type(value).__name__
    if actual not in allowed:
        mode = "final choice fusion" if final_choice_fusion else "normal execution"
        raise TypeError(f"{operator} {mode} expected {sorted(allowed)}, got {actual}")
