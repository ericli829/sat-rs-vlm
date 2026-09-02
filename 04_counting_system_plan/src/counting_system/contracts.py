"""COUNT 算子输入角色，对齐 sat_rs_vlm.taskgraph.contracts（taskgraph-v1.1）。"""

from __future__ import annotations

from typing import Any

from .runtime import EntitySet, ImageRef, Region, SelectResult, unwrap_select_result

COUNT_IMAGE_TYPES = (ImageRef, Region)
COUNT_ENTITY_TYPES = (EntitySet, SelectResult)


def validate_count_inputs(inputs: dict[str, Any]) -> None:
    keys = set(inputs)
    unexpected = keys - {"image", "entities"}
    if unexpected:
        raise ValueError(f"COUNT has unexpected input role(s): {sorted(unexpected)}")
    present = keys.intersection({"image", "entities"})
    if len(present) != 1:
        raise ValueError(f"COUNT requires exactly one of ['entities', 'image'], got {sorted(present)}")
    for role, value in inputs.items():
        if isinstance(value, list):
            raise TypeError(f"COUNT.{role} must be a single object, not a list")


def resolve_count_scope(inputs: dict[str, Any]) -> ImageRef | Region | EntitySet:
    validate_count_inputs(inputs)
    if "entities" in inputs:
        entities = unwrap_select_result(
            inputs["entities"],
            allow_empty=True,
            consumer="COUNT.entities",
        )
        if not isinstance(entities, EntitySet):
            raise TypeError("COUNT.entities must be EntitySet")
        return entities
    scope = inputs["image"]
    if not isinstance(scope, COUNT_IMAGE_TYPES):
        raise TypeError("COUNT.image must be ImageRef or Region")
    return scope
