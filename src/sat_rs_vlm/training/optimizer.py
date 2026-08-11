"""Non-overlapping optimizer parameter groups for H1 visual adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sat_rs_vlm.training.config import OptimizationGroupConfig


def build_h1_parameter_groups(
    model: Any,
    audit: Mapping[str, Any],
    learning_rates: OptimizationGroupConfig,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Map audited LoRA, merger, and ViT tensors to distinct learning rates.

    Every trainable tensor must appear exactly once. The audit is treated as the
    contract, making accidental optimizer omissions or duplicate membership fatal.
    """

    named_parameters = dict(model.named_parameters())

    def names_for(category: str) -> list[str]:
        payload = audit.get(category, {})
        if not isinstance(payload, Mapping):
            return []
        return [str(name) for name in payload.get("names", [])]

    lora_names = names_for("lora")
    merger_names = names_for("visual_merger")
    vision_names = names_for("vision_blocks")
    for name in names_for("optional_visual"):
        if "merger" in name.lower():
            merger_names.append(name)
        else:
            vision_names.append(name)
    grouped_names = {
        "lora": lora_names,
        "visual_merger": merger_names,
        "vision_blocks": vision_names,
    }
    learning_rate_by_group = {
        "lora": learning_rates.lora_lr,
        "visual_merger": learning_rates.visual_merger_lr,
        "vision_blocks": learning_rates.vision_lr,
    }
    seen_parameters: dict[int, str] = {}
    groups: list[dict[str, Any]] = []
    for group_name in ("lora", "visual_merger", "vision_blocks"):
        names = grouped_names[group_name]
        parameters: list[Any] = []
        for name in names:
            parameter = named_parameters.get(name)
            if parameter is None:
                raise ValueError(f"Audited optimizer parameter no longer exists: {name}")
            if not bool(parameter.requires_grad):
                raise ValueError(f"Audited optimizer parameter is no longer trainable: {name}")
            identity = id(parameter)
            previous = seen_parameters.get(identity)
            if previous is not None:
                raise ValueError(
                    f"Parameter {name} belongs to multiple optimizer groups: "
                    f"{previous}, {group_name}"
                )
            seen_parameters[identity] = group_name
            parameters.append(parameter)
        if parameters:
            groups.append(
                {
                    "group_name": group_name,
                    "params": parameters,
                    "lr": learning_rate_by_group[group_name],
                    "weight_decay": weight_decay,
                }
            )

    trainable = {
        id(parameter): name
        for name, parameter in named_parameters.items()
        if bool(parameter.requires_grad)
    }
    missing = sorted(
        name for identity, name in trainable.items() if identity not in seen_parameters
    )
    if missing:
        raise ValueError(f"Trainable parameters are missing from optimizer groups: {missing}")
    if not groups:
        raise ValueError("No H1 optimizer parameter groups were created")
    return groups


def create_h1_optimizer(
    torch: Any,
    model: Any,
    audit: Mapping[str, Any],
    learning_rates: OptimizationGroupConfig,
    *,
    weight_decay: float,
) -> Any:
    """Create AdamW with H1 groups for direct injection into Transformers Trainer."""

    groups = build_h1_parameter_groups(
        model,
        audit,
        learning_rates,
        weight_decay=weight_decay,
    )
    return torch.optim.AdamW(groups)


def optimizer_group_report(groups: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-safe group names, rates, and parameter counts."""

    return [
        {
            "name": str(group.get("group_name", "unknown")),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
            "tensor_count": len(group["params"]),
        }
        for group in groups
    ]
