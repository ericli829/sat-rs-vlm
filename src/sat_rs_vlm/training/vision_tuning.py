"""Explicit partial Qwen3-VL vision tuning and trainable-parameter auditing."""

from __future__ import annotations

import json
import warnings
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.training.config import TrainableAuditConfig, VisionTuningConfig

VISUAL_SIDECAR_FILENAME = "visual_trainable_weights.safetensors"
LEGACY_VISUAL_SIDECAR_FILENAME = "h1_visual_weights.safetensors"
VISUAL_SIDECAR_MANIFEST = "visual_trainable_manifest.json"


def resolve_visual_module(model: Any) -> Any:
    """Find the Qwen3-VL visual module through common model/PEFT wrappers.

    A valid visual module must expose ``blocks``, ``merger``, ``patch_embed``, and
    ``deepstack_merger_list``. This structural check is stable across wrapper name
    prefixes and rejects an unrelated module merely containing "visual" in its name.
    """

    queue: deque[Any] = deque([model])
    visited: set[int] = set()
    while queue:
        candidate = queue.popleft()
        identity = id(candidate)
        if identity in visited:
            continue
        visited.add(identity)
        if all(
            hasattr(candidate, attribute)
            for attribute in ("blocks", "merger", "patch_embed", "deepstack_merger_list")
        ):
            return candidate
        for attribute in ("base_model", "model", "module", "visual"):
            child = getattr(candidate, attribute, None)
            if child is not None and id(child) not in visited:
                queue.append(child)
    raise ValueError(
        "Could not resolve Qwen3-VL visual module with blocks/merger/patch_embed/"
        "deepstack_merger_list"
    )


def is_lora_parameter(name: str) -> bool:
    """Return whether a PEFT parameter name belongs to a LoRA adapter tensor."""

    lowered = name.lower()
    return any(token in lowered for token in ("lora_a", "lora_b", "lora_embedding"))


def _parameter_ids(module: Any) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _set_module_trainable(module: Any, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _parameter_count(parameters: list[Any]) -> int:
    return sum(int(parameter.numel()) for parameter in parameters)


def configure_h1_trainable_parameters(
    model: Any,
    vision_config: VisionTuningConfig,
    audit_config: TrainableAuditConfig,
) -> dict[str, Any]:
    """Freeze the base model, then enable LoRA and configured visual parameters.

    The function must run after an existing PEFT adapter has been loaded with
    ``is_trainable=True``. It returns a complete audit and raises when the requested
    visual surface was not actually enabled.
    """

    named_parameters = list(model.named_parameters())
    for _, parameter in named_parameters:
        parameter.requires_grad = False
    lora_names: list[str] = []
    for name, parameter in named_parameters:
        if is_lora_parameter(name):
            parameter.requires_grad = True
            lora_names.append(name)
    if not lora_names:
        raise ValueError(
            "H1 requires a loaded trainable LoRA adapter, but no LoRA parameters were found"
        )

    visual = resolve_visual_module(model)
    blocks = list(visual.blocks)
    if vision_config.unfreeze_last_n_blocks > len(blocks):
        raise ValueError(
            "vision_tuning.unfreeze_last_n_blocks exceeds available visual blocks: "
            f"requested={vision_config.unfreeze_last_n_blocks}, available={len(blocks)}"
        )
    selected_indices: list[int] = []
    selected_block_ids: set[int] = set()
    merger_ids: set[int] = set()
    optional_visual_ids: set[int] = set()
    if vision_config.enabled:
        requested_surface = (
            vision_config.unfreeze_last_n_blocks > 0
            or vision_config.train_main_merger
            or vision_config.train_deepstack_mergers
            or vision_config.train_patch_embed
        )
        if not requested_surface:
            raise ValueError(
                "vision_tuning.enabled=true but no non-LoRA visual surface was requested"
            )
        start = len(blocks) - vision_config.unfreeze_last_n_blocks
        selected_indices = list(range(start, len(blocks)))
        for index in selected_indices:
            _set_module_trainable(blocks[index], True)
            selected_block_ids.update(_parameter_ids(blocks[index]))
        if vision_config.train_main_merger:
            _set_module_trainable(visual.merger, True)
            merger_ids.update(_parameter_ids(visual.merger))
        if vision_config.train_deepstack_mergers:
            _set_module_trainable(visual.deepstack_merger_list, True)
            optional_visual_ids.update(_parameter_ids(visual.deepstack_merger_list))
        if vision_config.train_patch_embed:
            _set_module_trainable(visual.patch_embed, True)
            optional_visual_ids.update(_parameter_ids(visual.patch_embed))

    audit = audit_trainable_parameters(
        model,
        selected_block_ids=selected_block_ids,
        selected_block_indices=selected_indices,
        merger_ids=merger_ids,
        optional_visual_ids=optional_visual_ids,
    )
    if (
        vision_config.enabled
        and vision_config.unfreeze_last_n_blocks > 0
        and int(audit["vision_blocks"]["parameter_count"]) <= 0
    ):
        raise ValueError("vision_tuning.enabled=true but no visual block parameters are trainable")
    if (
        vision_config.enabled
        and vision_config.train_main_merger
        and int(audit["visual_merger"]["parameter_count"]) <= 0
    ):
        raise ValueError("train_main_merger=true but no visual merger parameters are trainable")
    unexpected = list(audit["other_trainable"])
    if unexpected:
        message = f"Unexpected trainable base-model parameters: {unexpected}"
        if audit_config.fail_on_unexpected_trainable:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return audit


def audit_trainable_parameters(
    model: Any,
    *,
    selected_block_ids: set[int],
    selected_block_indices: list[int],
    merger_ids: set[int],
    optional_visual_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Classify every trainable parameter and report complete names and counts."""

    optional_ids = optional_visual_ids or set()
    categories: dict[str, list[tuple[str, Any]]] = {
        "lora": [],
        "vision_blocks": [],
        "visual_merger": [],
        "optional_visual": [],
    }
    unexpected: list[str] = []
    total_parameters = 0
    trainable_parameters = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameters += count
        if not bool(parameter.requires_grad):
            continue
        trainable_parameters += count
        identity = id(parameter)
        if is_lora_parameter(name):
            categories["lora"].append((name, parameter))
        elif identity in selected_block_ids:
            categories["vision_blocks"].append((name, parameter))
        elif identity in merger_ids:
            categories["visual_merger"].append((name, parameter))
        elif identity in optional_ids:
            categories["optional_visual"].append((name, parameter))
        else:
            unexpected.append(name)

    def category_payload(key: str) -> dict[str, Any]:
        values = categories[key]
        return {
            "parameter_count": _parameter_count([parameter for _, parameter in values]),
            "tensor_count": len(values),
            "names": [name for name, _ in values],
        }

    vision_payload = category_payload("vision_blocks")
    vision_payload["block_indices"] = selected_block_indices
    return {
        "schema_version": "1.0",
        "lora": category_payload("lora"),
        "vision_blocks": vision_payload,
        "visual_merger": category_payload("visual_merger"),
        "optional_visual": category_payload("optional_visual"),
        "other_trainable": sorted(unexpected),
        "total_trainable": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_ratio": (trainable_parameters / total_parameters if total_parameters else 0.0),
    }


def write_trainable_audit(audit: Mapping[str, Any], path: str | Path) -> Path:
    """Persist the pre-optimizer audit as formatted JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(audit), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def selected_visual_parameter_names(audit: Mapping[str, Any]) -> list[str]:
    """Return all trained non-adapter visual parameter names from an audit."""

    names: list[str] = []
    for key in ("vision_blocks", "visual_merger", "optional_visual"):
        category = audit.get(key, {})
        if isinstance(category, Mapping):
            names.extend(str(value) for value in category.get("names", []))
    return names


def save_visual_sidecar(
    model: Any,
    audit: Mapping[str, Any],
    output_dir: str | Path,
    *,
    base_checkpoint: str | None = None,
    adapter_checkpoint: str | None = None,
    vision_tuning: Mapping[str, Any] | None = None,
) -> Path:
    """Save trained base visual tensors beside the PEFT adapter in safetensors."""

    names = set(selected_visual_parameter_names(audit))
    if not names:
        raise ValueError("No trained visual parameters are available for H1 sidecar saving")
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name in names
    }
    missing = sorted(names.difference(state))
    if missing:
        raise ValueError(f"Could not resolve audited visual tensors for saving: {missing}")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - model dependency environment
        raise ImportError("safetensors is required to save H1 visual weights") from exc
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / VISUAL_SIDECAR_FILENAME
    save_file(state, str(destination))
    manifest = {
        "schema_version": "1.0",
        "weights": destination.name,
        "sha256": file_sha256(destination),
        "selected_vit_blocks": list(audit.get("vision_blocks", {}).get("block_indices", [])),
        "parameter_names": sorted(state),
        "includes_main_merger": bool(audit.get("visual_merger", {}).get("names", [])),
        "includes_optional_visual": bool(audit.get("optional_visual", {}).get("names", [])),
        "base_checkpoint": base_checkpoint,
        "adapter_checkpoint": adapter_checkpoint,
        "vision_tuning": dict(vision_tuning or {}),
    }
    (root / VISUAL_SIDECAR_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_visual_sidecar(model: Any, path: str | Path) -> list[str]:
    """Load an H1 visual sidecar into a PEFT-wrapped model by audited names."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - model dependency environment
        raise ImportError("safetensors is required to load H1 visual weights") from exc
    source = Path(path)
    if source.is_dir():
        preferred = source / VISUAL_SIDECAR_FILENAME
        legacy = source / LEGACY_VISUAL_SIDECAR_FILENAME
        source = preferred if preferred.is_file() else legacy
    if not source.is_file():
        raise FileNotFoundError(f"Visual sidecar does not exist: {source}")
    state = load_file(str(source), device="cpu")
    current = dict(model.named_parameters())
    missing = sorted(set(state).difference(current))
    if missing:
        raise ValueError(f"H1 visual sidecar does not match the loaded model: {missing}")
    for name, tensor in state.items():
        parameter = current[name]
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return sorted(state)
