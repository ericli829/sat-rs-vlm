"""Training/runtime utilities for task-specialized merger experts."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.rs_merger_expert import (
    BASE_EXPERT,
    COUNTING_EXPERT,
    RSMergerExpertController,
    rs_detail_parameter_count,
)
from sat_rs_vlm.training.count_aware_loss import (
    EarlyLayerFeatureTap,
    auxiliary_ramp,
    categorical_count_loss,
    inverse_sqrt_class_weights,
    negative_binomial_count_loss,
)
from sat_rs_vlm.training.vision_tuning import load_visual_sidecar, resolve_visual_module


@dataclass(frozen=True)
class EffectiveEpochPlan:
    train_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch: int
    optimizer_steps_per_epoch: int
    resolved_max_steps: int
    expected_effective_epochs: float


@dataclass(frozen=True)
class ExpertCheckpointResume:
    root: Path
    weights: Path
    weights_sha256: str
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    training_state_path: Path | None
    completed_effective_epochs: float

    def report(self) -> dict[str, Any]:
        has_training_state = self.training_state_path is not None
        return {
            "source": self.root.as_posix(),
            "expert_weights": self.weights.as_posix(),
            "expert_weights_sha256": self.weights_sha256,
            "expert_manifest": self.manifest_path.as_posix(),
            "expert_manifest_sha256": self.manifest_sha256,
            "training_state": (
                self.training_state_path.as_posix() if self.training_state_path else None
            ),
            "continuation_mode": (
                "exact optimizer/scheduler continuation"
                if has_training_state
                else "weight continuation with fresh optimizer"
            ),
            "parent_completed_effective_epochs": self.completed_effective_epochs,
            "expert_weights_restored": True,
            "optimizer_state_restored": has_training_state,
            "scheduler_state_restored": has_training_state,
        }


def resolve_effective_epoch_plan(
    train_size: int,
    *,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
    target_effective_epochs: float = 1.0,
    max_steps: int | None = None,
) -> EffectiveEpochPlan:
    values = (train_size, per_device_batch_size, gradient_accumulation_steps, world_size)
    if any(value <= 0 for value in values):
        raise ValueError("train size, batch, accumulation, and world size must be positive")
    if target_effective_epochs <= 0:
        raise ValueError("target_effective_epochs must be positive")
    effective_batch = per_device_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(train_size / effective_batch)
    resolved = (
        int(max_steps)
        if max_steps is not None
        else math.ceil(steps_per_epoch * target_effective_epochs)
    )
    if resolved <= 0:
        raise ValueError("resolved max_steps must be positive")
    return EffectiveEpochPlan(
        train_size=train_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        effective_batch=effective_batch,
        optimizer_steps_per_epoch=steps_per_epoch,
        resolved_max_steps=resolved,
        expected_effective_epochs=resolved / steps_per_epoch,
    )


def _accumulation_window_size(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> int:
    """Return the real size of the current window, including an epoch tail."""

    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be within [1, {total_batches}], got {batch_index}")
    window_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - window_start)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
        if key not in {"task_types", "sample_ids", "auxiliary_counts"}
    }


def _shutdown_dataloader(loader: Any) -> None:
    """Stop persistent workers eagerly instead of waiting for cyclic GC."""

    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None


def cuda_memory_snapshot(device: torch.device | None = None) -> dict[str, int | bool]:
    """Report allocator counters without treating reserved cache as live tensors."""

    available = bool(torch.cuda.is_available())
    if not available:
        return {"cuda_available": False, "allocated_bytes": 0, "reserved_bytes": 0}
    target = device if device is not None and device.type == "cuda" else torch.device("cuda")
    return {
        "cuda_available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated(target)),
        "reserved_bytes": int(torch.cuda.memory_reserved(target)),
    }


def cuda_peak_memory_snapshot(device: torch.device | None = None) -> dict[str, int | bool]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "allocated_bytes": 0, "reserved_bytes": 0}
    target = device if device is not None and device.type == "cuda" else torch.device("cuda")
    return {
        "cuda_available": True,
        "allocated_bytes": int(torch.cuda.max_memory_allocated(target)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(target)),
    }


def _tensor_error(first: Tensor, second: Tensor) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError(f"Parity tensor shape mismatch: {first.shape} vs {second.shape}")
    delta = (first.float() - second.float()).abs()
    return {"max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item())}


def _parity_snapshot(
    model: nn.Module, batch: Mapping[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    visual = resolve_visual_module(model)
    visual_outputs: dict[str, Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> None:
            visual_outputs[name] = output.detach().float().cpu()

        return hook

    handles.append(visual.merger.register_forward_hook(capture("final")))
    for index, merger in enumerate(visual.deepstack_merger_list):
        handles.append(merger.register_forward_hook(capture(f"deepstack_{index}")))
    try:
        model.eval()
        device = next(model.parameters()).device
        inputs = _move_batch(batch, device)
        labels = inputs.pop("labels", None)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().float().cpu()
            generation_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            generated = (
                model.generate(
                    **generation_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                .detach()
                .cpu()
            )
        return {
            "logits": logits,
            "generation": generated,
            "visual": visual_outputs,
            "labels": labels.detach().cpu() if labels is not None else None,
        }
    finally:
        for handle in handles:
            handle.remove()


def capture_parity_snapshot(
    model: nn.Module, batch: Mapping[str, Any], *, max_new_tokens: int = 8
) -> dict[str, Any]:
    """Capture a fixed probe before or after route installation."""

    return _parity_snapshot(model, batch, max_new_tokens)


def compare_parity_snapshots(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    visual_keys_equal = set(first["visual"]) == set(second["visual"])
    report = {
        "generation_equal": torch.equal(first["generation"], second["generation"]),
        "logits_close": torch.allclose(first["logits"], second["logits"], atol=atol, rtol=rtol),
        "visual_outputs_close": visual_keys_equal
        and all(
            torch.allclose(first["visual"][name], second["visual"][name], atol=atol, rtol=rtol)
            for name in first["visual"]
        ),
        "logits_error": _tensor_error(first["logits"], second["logits"]),
        "visual_errors": {
            name: _tensor_error(value, second["visual"][name])
            for name, value in first["visual"].items()
            if name in second["visual"]
        },
        "atol": atol,
        "rtol": rtol,
    }
    report["passed"] = bool(
        report["generation_equal"] and report["logits_close"] and report["visual_outputs_close"]
    )
    return report


def merge_r1_with_parity(
    peft_model: nn.Module,
    probe_batch: Mapping[str, Any],
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    max_new_tokens: int = 8,
) -> tuple[nn.Module, dict[str, Any]]:
    """Capture PEFT behavior, merge exactly once, and prove parity."""

    if bool(getattr(peft_model, "_rs_r1_merged", False)):
        raise ValueError("R1 adapter has already been merged; refusing duplicate merge")
    merge = getattr(peft_model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("Configured R1 merge strategy requires a PEFT model with merge_and_unload")
    before = _parity_snapshot(peft_model, probe_batch, max_new_tokens)
    merged = merge(safe_merge=True)
    merged._rs_r1_merged = True
    after = _parity_snapshot(merged, probe_batch, max_new_tokens)
    logit_error = _tensor_error(before["logits"], after["logits"])
    visual_errors = {
        name: _tensor_error(value, after["visual"][name])
        for name, value in before["visual"].items()
        if name in after["visual"]
    }
    visual_keys_equal = set(before["visual"]) == set(after["visual"])
    generation_equal = torch.equal(before["generation"], after["generation"])
    logits_close = torch.allclose(before["logits"], after["logits"], atol=atol, rtol=rtol)
    visual_close = visual_keys_equal and all(
        torch.allclose(before["visual"][name], after["visual"][name], atol=atol, rtol=rtol)
        for name in before["visual"]
    )
    report = {
        "generation_equal": generation_equal,
        "logits_close": logits_close,
        "visual_outputs_close": visual_close,
        "logits_error": logit_error,
        "visual_errors": visual_errors,
        "atol": atol,
        "rtol": rtol,
        "passed": generation_equal and logits_close and visual_close,
    }
    if not report["passed"]:
        raise ValueError(f"R1 merge_and_unload parity failed: {report}")
    return merged, report


def expert_step0_parity(
    model: nn.Module,
    controller: RSMergerExpertController,
    probe_batch: Mapping[str, Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Prove that a newly attached C1/C2/C3 counting route equals base R1."""

    with controller.activate(BASE_EXPERT):
        baseline = _parity_snapshot(model, probe_batch, max_new_tokens)
    with controller.activate(COUNTING_EXPERT):
        expert = _parity_snapshot(model, probe_batch, max_new_tokens)
    logit_error = _tensor_error(baseline["logits"], expert["logits"])
    visual_errors = {
        name: _tensor_error(value, expert["visual"][name])
        for name, value in baseline["visual"].items()
        if name in expert["visual"]
    }
    report = {
        "generation_equal": torch.equal(baseline["generation"], expert["generation"]),
        "logits_close": torch.allclose(baseline["logits"], expert["logits"], atol=atol, rtol=rtol),
        "visual_outputs_close": set(baseline["visual"]) == set(expert["visual"])
        and all(
            torch.allclose(baseline["visual"][name], expert["visual"][name], atol=atol, rtol=rtol)
            for name in baseline["visual"]
        ),
        "logits_error": logit_error,
        "visual_errors": visual_errors,
        "atol": atol,
        "rtol": rtol,
    }
    report["passed"] = bool(
        report["generation_equal"] and report["logits_close"] and report["visual_outputs_close"]
    )
    if not report["passed"]:
        raise ValueError(f"Expert step-0 parity failed: {report}")
    return report


def load_r1_foundation(
    *,
    modules: dict[str, Any],
    base_model: str,
    r1_checkpoint: str,
    visual_sidecar: str,
    model_kwargs: Mapping[str, Any],
    processor_kwargs: Mapping[str, Any] | None = None,
    integration: str = "additive",
    probe_batch: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load R1 in the only valid order: LoRA, sidecar, optional parity-proven merge."""

    if integration not in {"additive", "merge"}:
        raise ValueError("r1 integration must be additive or merge")
    model, processor = load_qwen3vl(
        modules=modules,
        base_model=base_model,
        processor_source=base_model,
        model_kwargs=dict(model_kwargs),
        processor_kwargs=dict(processor_kwargs or {}),
        adapter_path=r1_checkpoint,
    )
    sidecar_names = load_visual_sidecar(model, visual_sidecar)
    parity: dict[str, Any] | None = None
    fallback_reason: str | None = None
    if integration == "merge":
        if probe_batch is None:
            raise ValueError("R1 merge requires a fixed probe batch for parity")
        try:
            model, parity = merge_r1_with_parity(model, probe_batch)
        except (TypeError, ValueError, RuntimeError) as exc:
            # A failed/missing merge is not guessed through. Reload one clean R1 copy and
            # retain its frozen PEFT path; the explicit exact-path Count LoRA is additive.
            fallback_reason = str(exc)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, processor = load_qwen3vl(
                modules=modules,
                base_model=base_model,
                processor_source=base_model,
                model_kwargs=dict(model_kwargs),
                processor_kwargs=dict(processor_kwargs or {}),
                adapter_path=r1_checkpoint,
            )
            sidecar_names = load_visual_sidecar(model, visual_sidecar)
            integration = "additive"
    for parameter in model.parameters():
        parameter.requires_grad = False
    return (
        model,
        processor,
        {
            "implementation": "merge_and_unload"
            if integration == "merge"
            else "frozen_peft_additive",
            "merge_fallback_reason": fallback_reason,
            "r1_loaded_once": True,
            "visual_sidecar_loaded_before_expert": True,
            "visual_sidecar_parameter_names": sidecar_names,
            "merge_parity": parity,
        },
    )


def parameter_grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared += float(parameter.grad.detach().float().norm(2).item()) ** 2
    return math.sqrt(squared)


def parameter_value_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = sum(float(parameter.detach().float().norm(2).item()) ** 2 for parameter in parameters)
    return math.sqrt(squared)


def inspect_expert_checkpoint(checkpoint: str | Path) -> ExpertCheckpointResume:
    """Inspect an expert sidecar and verify every declared local asset hash."""

    root = Path(checkpoint)
    if root.is_file():
        root = root.parent
    manifest_path = root / "expert_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Resume expert manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = root / str(manifest.get("expert_weights", "expert_model.safetensors"))
    if not weights.is_file():
        raise FileNotFoundError(f"Resume expert weights are missing: {weights}")
    weights_sha = file_sha256(weights)
    declared_weights_sha = manifest.get("expert_weights_sha256")
    if declared_weights_sha != weights_sha:
        raise ValueError(
            "Resume expert weight SHA256 does not match its manifest: "
            f"declared={declared_weights_sha}, actual={weights_sha}"
        )
    state_name = manifest.get("training_state")
    state_path = root / str(state_name) if state_name else root / "training_state.pt"
    if not state_path.is_file():
        state_path = None
    elif manifest.get("training_state_sha256") not in (None, file_sha256(state_path)):
        raise ValueError("Resume training state SHA256 does not match its manifest")
    training_summary_path = root / "training_summary.json"
    completed_effective_epochs = 0.0
    if training_summary_path.is_file():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        completed_effective_epochs = float(
            training_summary.get(
                "completed_effective_epochs",
                training_summary.get("plan", {}).get("expected_effective_epochs", 0.0),
            )
        )
    return ExpertCheckpointResume(
        root=root,
        weights=weights,
        weights_sha256=weights_sha,
        manifest=dict(manifest),
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        training_state_path=state_path,
        completed_effective_epochs=completed_effective_epochs,
    )


def validate_expert_checkpoint_compatibility(
    resume: ExpertCheckpointResume,
    *,
    expected_variant: str,
    expected_expert_variant: str,
    expected_detail_hidden_size: int,
    expected_local_depth: int,
    expected_interface_lora: Mapping[str, Any],
    architecture: Mapping[str, Any],
    architecture_audit_sha256: str,
    source_r1_manifest_sha256: str,
    source_visual_sidecar_sha256: str,
    source_r1_checkpoint: str,
    r1_integration: str,
) -> dict[str, Any]:
    """Fail closed on architecture, R1, visual-sidecar, and source-audit drift."""

    manifest = resume.manifest
    expected = {
        "variant": expected_variant,
        "selected_vit_blocks": list(architecture["deepstack_visual_indexes"])
        + [int(architecture["vision_block_count"]) - 1],
        "spatial_merge_size": int(architecture["spatial_merge_size"]),
        "visual_hidden_size": int(architecture["vision_hidden_size"]),
        "llm_hidden_size": int(architecture["llm_hidden_size"]),
        "architecture_audit_sha256": architecture_audit_sha256,
        "source_r1_manifest_sha256": source_r1_manifest_sha256,
        "source_visual_sidecar_sha256": source_visual_sidecar_sha256,
        "source_r1_checkpoint": source_r1_checkpoint,
        "r1_integration": r1_integration,
    }
    mismatches = [
        f"{key}: checkpoint={manifest.get(key)!r}, runtime={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    legacy_inferred: list[str] = []
    expected_merger_parameter_count = len(expected["selected_vit_blocks"]) * (
        rs_detail_parameter_count(
            int(architecture["vision_hidden_size"]),
            int(architecture["llm_hidden_size"]),
            int(expected_detail_hidden_size),
            local_depth=int(expected_local_depth),
            spatial_merge_size=int(architecture["spatial_merge_size"]),
        )
    )
    for key, value in (
        ("expert_variant", expected_expert_variant),
        ("detail_hidden_size", int(expected_detail_hidden_size)),
        ("local_depth", int(expected_local_depth)),
    ):
        if key not in manifest:
            # Schema 1.0 predates these fields. Fail closed unless its stored parameter
            # count proves the same rs_detail architecture; exact tensor shapes are then
            # checked by controller.load_expert_state_dict.
            if (
                expected_expert_variant != "rs_detail"
                or int(manifest.get("merger_parameter_count", -1))
                != expected_merger_parameter_count
            ):
                mismatches.append(
                    f"{key}: missing and legacy merger_parameter_count does not prove "
                    f"the runtime rs_detail architecture ({expected_merger_parameter_count})"
                )
            else:
                legacy_inferred.append(key)
        elif manifest.get(key) != value:
            mismatches.append(f"{key}: checkpoint={manifest.get(key)!r}, runtime={value!r}")
    checkpoint_lora = manifest.get("interface_lora")
    if not isinstance(checkpoint_lora, Mapping):
        mismatches.append("interface_lora: checkpoint manifest is missing the mapping")
    else:
        for key in ("enabled", "layers", "targets", "r", "alpha", "dropout"):
            expected_value = expected_interface_lora.get(key)
            if checkpoint_lora.get(key) != expected_value:
                mismatches.append(
                    f"interface_lora.{key}: checkpoint={checkpoint_lora.get(key)!r}, "
                    f"runtime={expected_value!r}"
                )
    if mismatches:
        raise ValueError("Composite checkpoint compatibility mismatch: " + "; ".join(mismatches))
    return {
        **resume.report(),
        "validated_variant": expected_variant,
        "validated_expert_variant": expected_expert_variant,
        "validated_detail_hidden_size": int(expected_detail_hidden_size),
        "validated_local_depth": int(expected_local_depth),
        "legacy_inferred_manifest_fields": legacy_inferred,
    }


def load_expert_checkpoint(
    controller: RSMergerExpertController,
    resume: ExpertCheckpointResume,
) -> None:
    """Restore exact expert tensors after compatibility validation has passed."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required for composite expert checkpoints") from exc
    controller.load_expert_state_dict(load_file(str(resume.weights), device="cpu"))


def restore_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Restore optional full continuation state and return progress metadata."""

    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return {
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "global_optimizer_step": int(payload.get("global_optimizer_step", 0)),
        "completed_effective_epochs": float(payload.get("completed_effective_epochs", 0.0)),
    }


def save_composite_checkpoint(
    controller: RSMergerExpertController,
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    training_state_path: str | Path | None = None,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("safetensors is required for composite expert checkpoints") from exc
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in controller.expert_state_dict().items()
    }
    weights = root / "expert_model.safetensors"
    save_file(state, str(weights))
    payload = dict(manifest)
    payload["expert_weights"] = weights.name
    payload["expert_weights_sha256"] = file_sha256(weights)
    if training_state_path is not None:
        source_state = Path(training_state_path)
        if not source_state.is_file():
            raise FileNotFoundError(f"Training state is missing: {source_state}")
        destination_state = root / "training_state.pt"
        shutil.copy2(source_state, destination_state)
        payload["training_state"] = destination_state.name
        payload["training_state_sha256"] = file_sha256(destination_state)
    (root / "expert_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "training_summary.json").write_text(
        json.dumps(dict(training_summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    import yaml

    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump(dict(resolved_config), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return root


def _save_epoch_training_checkpoint(
    controller: RSMergerExpertController,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    epoch: int,
    global_optimizer_step: int,
    completed_effective_epochs: float,
) -> Path:
    """Persist a restartable sidecar at every completed effective epoch."""

    from safetensors.torch import save_file

    destination = output_dir / "epoch_checkpoints" / f"epoch_{epoch:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in controller.expert_state_dict().items()
    }
    save_file(state, str(destination / "expert_model.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_optimizer_step": global_optimizer_step,
            "completed_effective_epochs": completed_effective_epochs,
        },
        destination / "training_state.pt",
    )
    (destination / "epoch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "epoch": epoch,
                "global_optimizer_step": global_optimizer_step,
                "completed_effective_epochs": completed_effective_epochs,
                "evaluation_status": "pending_fixed_e_count_v2_evaluation",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def validate_checkpoint_provenance(
    manifest: Mapping[str, Any],
    *,
    architecture_audit_sha256: str,
    source_r1_manifest_sha256: str,
    source_visual_sidecar_sha256: str,
) -> None:
    expected = {
        "architecture_audit_sha256": architecture_audit_sha256,
        "source_r1_manifest_sha256": source_r1_manifest_sha256,
        "source_visual_sidecar_sha256": source_visual_sidecar_sha256,
    }
    mismatches = [
        f"{key}: checkpoint={manifest.get(key)!r}, runtime={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError("Composite checkpoint provenance mismatch: " + "; ".join(mismatches))


def run_training(
    *,
    model: nn.Module,
    processor: Any,
    controller: RSMergerExpertController,
    train_file: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    training_config: Mapping[str, Any],
    max_train_samples: int | None = None,
    max_steps: int | None = None,
    forward_only: bool = False,
    resume_training_state_path: str | Path | None = None,
    resume_completed_effective_epochs: float = 0.0,
) -> dict[str, Any]:
    """Small explicit loop: assistant-only model CE is the sole objective."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    memory_log_path = output / "memory_log.json"
    device = next(model.parameters()).device
    memory_log: dict[str, Any] = {
        "before_train": cuda_memory_snapshot(device),
        "reserved_memory_note": (
            "reserved_bytes is allocator cache, not live tensor allocation; "
            "allocated_bytes is the live allocator counter"
        ),
    }
    result: dict[str, Any] | None = None
    dataset = collator = generator = loader = None
    optimizer = scheduler = None
    merger_parameters: list[nn.Parameter] = []
    lora_parameters: list[nn.Parameter] = []
    count_head_parameters: list[nn.Parameter] = []
    feature_tap: EarlyLayerFeatureTap | None = None
    class_weights: Tensor | None = None
    batch = values = outputs = loss = row = task_types = all_trainable = None
    sample_ids = auxiliary_counts = count_result = count_logits = count_features = None
    lm_loss = count_targets = None
    initial_expert = final_state = None
    try:
        dataset = Qwen3VLDataset(train_file, max_samples=max_train_samples)
        if not dataset:
            raise ValueError("Training dataset is empty")
        wrong_tasks = sorted(
            {row["task_type"] for row in dataset if row["task_type"] != "counting"}
        )
        if wrong_tasks:
            raise ValueError(f"Counting expert training received non-counting tasks: {wrong_tasks}")
        batch_size = int(training_config.get("per_device_train_batch_size", 4))
        accumulation = int(training_config.get("gradient_accumulation_steps", 4))
        plan = resolve_effective_epoch_plan(
            len(dataset),
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            target_effective_epochs=float(training_config.get("target_effective_epochs", 1.0)),
            max_steps=max_steps,
        )
        audit = controller.freeze_base_and_enable_expert()
        controller.set_active_expert(COUNTING_EXPERT)
        count_config = dict(training_config.get("count_loss", {}))
        count_loss_enabled = bool(count_config.get("enabled", False))
        max_count = int(count_config.get("max_count", 15))
        if count_loss_enabled:
            if controller.count_head is None:
                raise ValueError("count_loss.enabled requires controller.configure_count_head()")
            if str(count_config.get("tail_policy", "clip_to_K")) != "clip_to_K":
                raise ValueError("The only supported count tail policy is clip_to_K")
            distribution = str(count_config.get("distribution", "categorical"))
            if distribution not in {"categorical", "negative_binomial"}:
                raise ValueError(f"Unsupported count distribution: {distribution}")
            frequencies = [0] * (max_count + 1)
            for sample in dataset:
                target = Qwen3VLDataCollator._auxiliary_count_target(sample)
                if target >= 0:
                    frequencies[min(target, max_count)] += 1
            class_weights = inverse_sqrt_class_weights(frequencies)
            feature_tap = EarlyLayerFeatureTap(
                model, layer_index=int(count_config.get("feature_layer", 3))
            )
        collator = Qwen3VLDataCollator(
            processor,
            int(training_config.get("max_seq_length", 2048)),
            image_root,
            include_task_metadata=True,
        )
        generator = torch.Generator()
        generator.manual_seed(int(training_config.get("seed", 42)))
        worker_count = int(training_config.get("dataloader_num_workers", 8))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=worker_count,
            pin_memory=bool(training_config.get("dataloader_pin_memory", True)),
            persistent_workers=bool(training_config.get("dataloader_persistent_workers", True))
            and worker_count > 0,
            generator=generator,
        )
        merger_parameters = [
            parameter
            for merger in controller.routed_mergers
            for _, parameter in merger.expert_named_parameters("expert")
            if parameter.requires_grad
        ]
        lora_parameters = [
            parameter
            for module in controller.interface_modules
            for name, parameter in module.named_parameters()
            if name.startswith("lora_") and parameter.requires_grad
        ]
        count_head_parameters = (
            [
                parameter
                for parameter in controller.count_head.parameters()
                if parameter.requires_grad
            ]
            if controller.count_head is not None
            else []
        )
        groups = [
            {
                "params": merger_parameters,
                "lr": float(training_config["merger_lr"]),
                "weight_decay": float(training_config.get("merger_weight_decay", 0.01)),
                "group_name": "merger",
            }
        ]
        if lora_parameters:
            groups.append(
                {
                    "params": lora_parameters,
                    "lr": float(training_config.get("interface_lora_lr", 2e-5)),
                    "weight_decay": 0.0,
                    "group_name": "interface_lora",
                }
            )
        if count_head_parameters:
            groups.append(
                {
                    "params": count_head_parameters,
                    "lr": float(training_config.get("count_head_lr", 2e-4)),
                    "weight_decay": float(training_config.get("count_head_weight_decay", 0.01)),
                    "group_name": "count_head",
                }
            )
        optimizer = torch.optim.AdamW(groups)
        warmup_steps = math.ceil(
            plan.resolved_max_steps * float(training_config.get("warmup_ratio", 0.03))
        )
        scheduler_name = str(training_config.get("scheduler", "constant_after_warmup"))
        if scheduler_name not in {"constant_after_warmup", "cosine"}:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        def schedule(step: int) -> float:
            warmup = min(1.0, (step + 1) / max(1, warmup_steps))
            if step < warmup_steps:
                return warmup
            if scheduler_name == "constant_after_warmup":
                return 1.0
            progress = (step - warmup_steps) / max(1, plan.resolved_max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        resume_progress = {
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "global_optimizer_step": 0,
            "completed_effective_epochs": float(resume_completed_effective_epochs),
        }
        if resume_training_state_path is not None:
            resume_progress = restore_training_state(
                optimizer, scheduler, resume_training_state_path, device=device
            )
        if bool(training_config.get("gradient_checkpointing", True)):
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if callable(enable):
                try:
                    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError as exc:
                    enable()
                    if count_loss_enabled:
                        raise RuntimeError(
                            "count-aware hooks require non-reentrant gradient checkpointing"
                        ) from exc
        controller.set_training_mode(True)
        # Eval mode keeps foundation dropout disabled without disconnecting expert gradients.
        log_path = output / "train_log.jsonl"
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optimizer_step = 0
        last_checkpoint_epoch = int(
            math.floor(float(resume_progress["completed_effective_epochs"]) + 1e-9)
        )
        samples = 0
        last_loss = None
        forward_complete = False
        initial_expert = {
            name: value.detach().cpu().clone()
            for name, value in controller.expert_state_dict().items()
        }
        while optimizer_step < plan.resolved_max_steps and not forward_complete:
            total_batches = len(loader)
            for batch_index, batch in enumerate(loader, 1):
                current_window_size = _accumulation_window_size(
                    batch_index, total_batches, accumulation
                )
                task_types = batch.pop("task_types")
                sample_ids = batch.pop("sample_ids")
                auxiliary_counts = batch.pop("auxiliary_counts")
                if set(task_types) != {"counting"}:
                    raise AssertionError(f"Training router received invalid tasks: {task_types}")
                values = _move_batch(batch, device)
                supervised_tokens = int((values["labels"] != -100).sum().item())
                if supervised_tokens <= 0:
                    raise ValueError("Assistant-only mask produced zero supervised tokens")
                autocast_enabled = device.type == "cuda" and bool(training_config.get("bf16", True))
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    outputs = model(**values)
                    lm_loss = outputs.loss
                    loss = lm_loss
                    count_result = None
                    if count_loss_enabled:
                        assert feature_tap is not None
                        assert controller.count_head is not None
                        count_features = feature_tap.take_anchors(
                            values["labels"], values.get("attention_mask")
                        )
                        count_logits = controller.count_head(count_features)
                        count_targets = torch.tensor(
                            auxiliary_counts, dtype=torch.long, device=device
                        )
                        effective_epoch = (
                            float(resume_progress["completed_effective_epochs"])
                            + (optimizer_step + 1) / plan.optimizer_steps_per_epoch
                        )
                        ramp = auxiliary_ramp(
                            effective_epoch,
                            ramp_epochs=float(count_config.get("ramp_epochs", 0.1)),
                        )
                        if distribution == "categorical":
                            count_result = categorical_count_loss(
                                count_logits,
                                count_targets,
                                max_count=max_count,
                                class_weights=class_weights,
                                epsilon=float(count_config.get("epsilon", 0.15)),
                                tau=float(count_config.get("tau", 1.0)),
                                classification_weight=float(
                                    count_config.get("classification_weight", 0.5)
                                ),
                                ordinal_weight=float(count_config.get("ordinal_weight", 1.0)),
                                regression_weight=float(
                                    count_config.get("regression_weight", 0.25)
                                ),
                                auxiliary_weight=ramp,
                            )
                        else:
                            count_result = negative_binomial_count_loss(
                                count_logits,
                                count_targets,
                                max_count=max_count,
                                class_weights=class_weights,
                                nll_weight=float(count_config.get("nb_nll_weight", 1.0)),
                                regression_weight=float(
                                    count_config.get("regression_weight", 0.25)
                                ),
                                auxiliary_weight=ramp,
                            )
                        loss = lm_loss + count_result.total
                if loss is None or not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite assistant-only LM loss: {loss}")
                last_loss = float(loss.detach().item())
                samples += len(task_types)
                if forward_only:
                    elapsed = time.perf_counter() - start
                    result = {
                        "mode": "forward_only",
                        "loss_total": last_loss,
                        "loss_lm": float(lm_loss.detach().item()),
                        "supervised_tokens": supervised_tokens,
                        "elapsed_seconds": elapsed,
                        "plan": asdict(plan),
                        "trainable_audit": audit,
                    }
                    forward_complete = True
                    break
                (loss / current_window_size).backward()
                should_step = batch_index % accumulation == 0 or batch_index == total_batches
                if not should_step:
                    continue
                all_trainable = merger_parameters + lora_parameters + count_head_parameters
                unclipped_global_grad = parameter_grad_norm(all_trainable)
                merger_grad = parameter_grad_norm(merger_parameters)
                lora_grad = parameter_grad_norm(lora_parameters)
                count_head_grad = parameter_grad_norm(count_head_parameters)
                clip_return = float(
                    torch.nn.utils.clip_grad_norm_(
                        all_trainable, float(training_config.get("max_grad_norm", 1.0))
                    ).item()
                )
                clipped_global_grad = parameter_grad_norm(all_trainable)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                global_step = int(resume_progress["global_optimizer_step"]) + optimizer_step
                elapsed = time.perf_counter() - start
                row = {
                    "loss_total": last_loss,
                    "loss_lm": float(lm_loss.detach().item()),
                    "supervised_tokens": supervised_tokens,
                    "grad_norm_unclipped": unclipped_global_grad,
                    "grad_norm_clip_return": clip_return,
                    "grad_norm_clipped": clipped_global_grad,
                    "merger_grad_norm_unclipped": merger_grad,
                    "interface_lora_grad_norm_unclipped": lora_grad,
                    "count_head_grad_norm_unclipped": count_head_grad,
                    "merger_parameter_norm": parameter_value_norm(merger_parameters),
                    "interface_lora_parameter_norm": parameter_value_norm(lora_parameters),
                    "count_head_parameter_norm": parameter_value_norm(count_head_parameters),
                    "learning_rates": {
                        group.get("group_name", str(index)): group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    },
                    "peak_vram_gb": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    ),
                    "samples_per_second": samples / max(elapsed, 1e-9),
                    "optimizer_step": optimizer_step,
                    "global_optimizer_step": global_step,
                    "accumulation_window_size": current_window_size,
                    "elapsed_seconds": elapsed,
                    "sample_ids": sample_ids,
                }
                if count_result is not None:
                    row.update(count_result.detached_log())
                    row["loss_total"] = last_loss
                    row["loss_lm"] = float(lm_loss.detach().item())
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed_now = float(resume_progress["completed_effective_epochs"]) + (
                    optimizer_step / plan.optimizer_steps_per_epoch
                )
                completed_integer_epoch = int(math.floor(completed_now + 1e-9))
                if completed_integer_epoch > last_checkpoint_epoch:
                    for completed_epoch in range(
                        last_checkpoint_epoch + 1, completed_integer_epoch + 1
                    ):
                        _save_epoch_training_checkpoint(
                            controller,
                            optimizer,
                            scheduler,
                            output,
                            epoch=completed_epoch,
                            global_optimizer_step=global_step,
                            completed_effective_epochs=completed_now,
                        )
                    last_checkpoint_epoch = completed_integer_epoch
                if optimizer_step >= plan.resolved_max_steps:
                    break
        if result is None:
            elapsed = time.perf_counter() - start
            final_state = {
                name: value.detach().cpu() for name, value in controller.expert_state_dict().items()
            }
            changed = [
                name
                for name in final_state
                if not torch.equal(final_state[name], initial_expert[name])
            ]
            if not changed:
                raise AssertionError("No expert parameter changed during training")
            lora_changed = [name for name in changed if name.startswith("interface_lora.")]
            if lora_parameters and not lora_changed:
                raise AssertionError("C3 interface LoRA parameters did not change")
            global_steps = int(resume_progress["global_optimizer_step"]) + optimizer_step
            completed_epochs = float(resume_progress["completed_effective_epochs"]) + float(
                plan.expected_effective_epochs
            )
            training_state_path = output / "training_state.pt"
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_optimizer_step": global_steps,
                    "completed_effective_epochs": completed_epochs,
                },
                training_state_path,
            )
            result = {
                "mode": "trained",
                "loss_total": last_loss,
                "optimizer_steps": optimizer_step,
                "global_optimizer_steps": global_steps,
                "continuation_effective_epochs": plan.expected_effective_epochs,
                "completed_effective_epochs": completed_epochs,
                "resume_training_state": resume_progress,
                "training_state_path": training_state_path.as_posix(),
                "elapsed_seconds": elapsed,
                "samples_per_second": samples / max(elapsed, 1e-9),
                "optimizer_steps_per_second": optimizer_step / max(elapsed, 1e-9),
                "peak_allocated_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0,
                "peak_reserved_vram_gb": torch.cuda.max_memory_reserved(device) / 1024**3
                if device.type == "cuda"
                else 0.0,
                "changed_expert_tensors": changed,
                "trainable_params": audit["total_trainable_parameter_count"],
                "base_params": audit["base_parameter_count"],
                "expert_params": audit["expert_parameter_count"],
                "lora_params": audit["interface_lora_parameter_count"],
                "plan": asdict(plan),
                "trainable_audit": audit,
            }
    finally:
        memory_log["peak"] = cuda_peak_memory_snapshot(device)
        memory_log["before_cleanup"] = cuda_memory_snapshot(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if loader is not None:
            _shutdown_dataloader(loader)
        if feature_tap is not None:
            feature_tap.close()
        for merger in controller.routed_mergers:
            merger.clear_runtime_state()
        controller.set_training_mode(False)
        batch = values = outputs = loss = row = None
        task_types = sample_ids = auxiliary_counts = all_trainable = None
        lm_loss = count_targets = None
        count_result = count_logits = count_features = class_weights = None
        initial_expert = final_state = None
        merger_parameters.clear()
        lora_parameters.clear()
        count_head_parameters.clear()
        optimizer = scheduler = loader = generator = collator = dataset = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        memory_log["after_cleanup"] = cuda_memory_snapshot(device)
        memory_log_path.write_text(
            json.dumps(memory_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if result is not None:
            result["memory"] = memory_log
            result["memory_log"] = memory_log_path.as_posix()
    if result is None:
        raise AssertionError("Training completed without producing a result")
    return result
