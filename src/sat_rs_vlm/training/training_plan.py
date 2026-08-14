"""Resolve auditable training-step budgets from effective epochs."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingPlan:
    unique_samples: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int
    steps_per_epoch: int
    resolved_max_steps: int | None
    expected_effective_epochs: float
    expected_sample_exposures: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", **asdict(self)}


def detected_world_size(environ: dict[str, str] | None = None) -> int:
    """Read distributed world size without importing torch/distributed."""

    value = (environ or os.environ).get("WORLD_SIZE", "1")
    try:
        world_size = int(value)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE must be an integer, got {value!r}") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    return world_size


def resolve_training_plan(
    *,
    unique_samples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    max_steps: int | None,
    num_train_epochs: float | None,
    target_effective_epochs: float | None,
    max_effective_epochs: float | None,
    allow_overtrain: bool,
) -> TrainingPlan:
    """Resolve steps and reject accidental excessive sample exposure."""

    if unique_samples < 1:
        raise ValueError("unique_samples must be positive")
    if min(per_device_batch_size, gradient_accumulation_steps, world_size) < 1:
        raise ValueError("batch size, gradient accumulation and world size must be positive")
    effective_batch = per_device_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(unique_samples / effective_batch)

    if max_steps is not None:
        resolved_steps = max_steps
        expected_epochs = max_steps / steps_per_epoch
        source = "explicit_max_steps"
    elif target_effective_epochs is not None:
        resolved_steps = max(1, math.ceil(steps_per_epoch * target_effective_epochs))
        expected_epochs = resolved_steps / steps_per_epoch
        source = "target_effective_epochs"
    elif num_train_epochs is not None:
        resolved_steps = None
        expected_epochs = float(num_train_epochs)
        source = "num_train_epochs"
    else:
        raise ValueError("Set max_steps, target_effective_epochs, or num_train_epochs for training")

    if (
        max_effective_epochs is not None
        and expected_epochs > max_effective_epochs + 1.0e-12
        and not allow_overtrain
    ):
        raise ValueError(
            "Resolved training budget exceeds max_effective_epochs: "
            f"expected={expected_epochs:.4f}, maximum={max_effective_epochs:.4f}. "
            "Set training.allow_overtrain=true only for an intentional override."
        )
    exposures = (
        min(unique_samples, resolved_steps * effective_batch)
        if resolved_steps is not None and expected_epochs <= 1.0
        else math.ceil(unique_samples * expected_epochs)
    )
    return TrainingPlan(
        unique_samples=unique_samples,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        effective_batch_size=effective_batch,
        steps_per_epoch=steps_per_epoch,
        resolved_max_steps=resolved_steps,
        expected_effective_epochs=expected_epochs,
        expected_sample_exposures=exposures,
        source=source,
    )
