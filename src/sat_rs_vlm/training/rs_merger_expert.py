"""Training/runtime utilities for task-specialized merger experts."""

from __future__ import annotations

import json
import math
import os
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
        if key != "task_types"
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
            "labels": labels,
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


def save_composite_checkpoint(
    controller: RSMergerExpertController,
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
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
) -> dict[str, Any]:
    """Small explicit loop: assistant-only model CE is the sole objective."""

    dataset = Qwen3VLDataset(train_file, max_samples=max_train_samples)
    if not dataset:
        raise ValueError("Training dataset is empty")
    wrong_tasks = sorted({row["task_type"] for row in dataset if row["task_type"] != "counting"})
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
    device = next(model.parameters()).device
    collator = Qwen3VLDataCollator(
        processor,
        int(training_config.get("max_seq_length", 2048)),
        image_root,
        include_task_metadata=True,
    )
    generator = torch.Generator()
    generator.manual_seed(int(training_config.get("seed", 42)))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=int(training_config.get("dataloader_num_workers", 8)),
        pin_memory=bool(training_config.get("dataloader_pin_memory", True)),
        persistent_workers=bool(training_config.get("dataloader_persistent_workers", True))
        and int(training_config.get("dataloader_num_workers", 8)) > 0,
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
    optimizer = torch.optim.AdamW(groups)
    warmup_steps = math.ceil(
        plan.resolved_max_steps * float(training_config.get("warmup_ratio", 0.03))
    )

    def schedule(step: int) -> float:
        return min(1.0, (step + 1) / max(1, warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if bool(training_config.get("gradient_checkpointing", True)):
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
    controller.set_training_mode(True)
    # Eval mode does not disable autograd: frozen LLM operations remain in the graph, so
    # assistant-only LM loss still reaches the visual expert without enabling foundation dropout.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer_step = 0
    samples = 0
    last_loss = None
    initial_expert = {
        name: value.detach().cpu().clone() for name, value in controller.expert_state_dict().items()
    }
    while optimizer_step < plan.resolved_max_steps:
        total_batches = len(loader)
        for batch_index, batch in enumerate(loader, 1):
            current_window_size = _accumulation_window_size(
                batch_index,
                total_batches,
                accumulation,
            )
            task_types = batch.pop("task_types")
            if set(task_types) != {"counting"}:
                raise AssertionError(f"Training router received invalid tasks: {task_types}")
            values = _move_batch(batch, device)
            supervised_tokens = int((values["labels"] != -100).sum().item())
            if supervised_tokens <= 0:
                raise ValueError("Assistant-only mask produced zero supervised tokens")
            autocast_enabled = device.type == "cuda" and bool(training_config.get("bf16", True))
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                outputs = model(**values)
                loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite assistant-only LM loss: {loss}")
            last_loss = float(loss.detach().item())
            samples += len(task_types)
            if forward_only:
                elapsed = time.perf_counter() - start
                return {
                    "mode": "forward_only",
                    "loss_total": last_loss,
                    "supervised_tokens": supervised_tokens,
                    "elapsed_seconds": elapsed,
                    "plan": asdict(plan),
                    "trainable_audit": audit,
                }
            (loss / current_window_size).backward()
            should_step = batch_index % accumulation == 0 or batch_index == total_batches
            if not should_step:
                continue
            all_trainable = merger_parameters + lora_parameters
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    all_trainable, float(training_config.get("max_grad_norm", 1.0))
                ).item()
            )
            merger_grad = parameter_grad_norm(merger_parameters)
            lora_grad = parameter_grad_norm(lora_parameters)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            elapsed = time.perf_counter() - start
            row = {
                "loss_total": last_loss,
                "supervised_tokens": supervised_tokens,
                "grad_norm": grad_norm,
                "merger_grad_norm": merger_grad,
                "interface_lora_grad_norm": lora_grad,
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
                "accumulation_window_size": current_window_size,
                "elapsed_seconds": elapsed,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if optimizer_step >= plan.resolved_max_steps:
                break
    elapsed = time.perf_counter() - start
    final_state = {
        name: value.detach().cpu() for name, value in controller.expert_state_dict().items()
    }
    changed = [
        name for name in final_state if not torch.equal(final_state[name], initial_expert[name])
    ]
    if not changed:
        raise AssertionError("No expert parameter changed during training")
    lora_changed = [name for name in changed if name.startswith("interface_lora.")]
    if lora_parameters and not lora_changed:
        raise AssertionError("C3 interface LoRA parameters did not change")
    return {
        "mode": "trained",
        "loss_total": last_loss,
        "optimizer_steps": optimizer_step,
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
