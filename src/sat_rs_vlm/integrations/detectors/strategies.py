"""One configurable strategy catalog for LOCATE and CountingSystem tile planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_detection_strategy(category: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    key = category.strip().casefold().replace("_", " ")
    mapping = config.get("category_map", {})
    if not isinstance(mapping, Mapping):
        raise ValueError("detection_strategies.category_map must be a mapping")
    name = str(mapping.get(key, config.get("default", "legacy")))
    if name in {"legacy", "default"}:
        return {"name": "legacy"}
    if name not in {"dense_small", "large_multi"}:
        raise ValueError(f"unknown detection strategy {name!r}")
    options = config.get(name, {})
    if not isinstance(options, Mapping):
        raise ValueError(f"detection_strategies.{name} must be a mapping")
    result = {
        "name": name,
        "score_threshold": 0.2,
        "overlap_ratio": 0.15,
        "spatial_resize_enabled": name == "large_multi",
        "tile_size": 800,
        "grid_size": 2,
        "max_refinement_regions": 4,
        "large_box_area_fraction": 0.20,
        **dict(options),
    }
    if not 0 <= float(result["score_threshold"]) <= 1:
        raise ValueError("DINO strategy score_threshold must be in [0, 1]")
    if not 0 <= float(result["overlap_ratio"]) < 1:
        raise ValueError("DINO strategy overlap_ratio must be in [0, 1)")
    if int(result["tile_size"]) < 1 or int(result["grid_size"]) != 2:
        raise ValueError("dense tile_size must be positive and large_multi grid_size must be 2")
    return result
