"""Pure-data analysis helpers for SEU sensitivity reports."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def summarize_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate completed condition rows without loading a model."""
    groups: dict[tuple[str, str, int | None], list[float]] = defaultdict(list)
    bit_indices: dict[int, list[float]] = defaultdict(list)
    projections: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        changed = row.get("changed_rate")
        if not isinstance(changed, (int, float)):
            continue
        layer = row.get("layers", [None])
        layer_value = layer[0] if layer else None
        groups[(str(row.get("target")), str(row.get("bit_plane")), layer_value)].append(float(changed))
        for record in row.get("records", []):
            index = record.get("bit_index")
            if isinstance(index, int):
                bit_indices[index].append(float(changed))
            name = str(record.get("target_name", "")).lower()
            projection = next((token for token in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if token in name), "other")
            projections[projection].append(float(changed))
    def stats(values: list[float]) -> dict[str, float]:
        return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0, "count": float(len(values))}
    return {
        "groups": {f"{target}|{plane}|{layer}": stats(values) for (target, plane, layer), values in groups.items()},
        "bit_indices": {str(index): stats(values) for index, values in bit_indices.items()},
        "projections": {name: stats(values) for name, values in projections.items()},
    }
