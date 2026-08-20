"""Qwen3-VL-4B Stage-A v2 runner 的纯 Python 规划与契约校验工具。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sat_rs_vlm.data.task_sampler import build_alternating_source_sampler
from sat_rs_vlm.models.reliability.checksum import file_sha256

R0_STAGE = "qwen3vl_4b_stage_a_v2_r0"
R1_STAGE = "qwen3vl_4b_stage_a_v2_r1"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class StageEpochPlan:
    """由冻结数据规模和 effective batch 推导的一轮训练计划。"""

    sample_count: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int
    steps_per_epoch: int
    half_checkpoint_step: int
    final_step: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def resolve_stage_epoch_plan(
    sample_count: int,
    *,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> StageEpochPlan:
    """动态解析一轮步数和约 0.5 epoch 保存点，不依赖固定 step 常量。"""

    values = (
        sample_count,
        per_device_batch_size,
        gradient_accumulation_steps,
        world_size,
    )
    if min(values) < 1:
        raise ValueError(
            "Stage sample count, batch size, accumulation, and world size " "must be positive"
        )
    effective_batch = per_device_batch_size * gradient_accumulation_steps * world_size
    steps = math.ceil(sample_count / effective_batch)
    return StageEpochPlan(
        sample_count=sample_count,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
        effective_batch_size=effective_batch,
        steps_per_epoch=steps,
        half_checkpoint_step=max(1, round(steps * 0.5)),
        final_step=steps,
    )


def adapter_is_complete(path: str | Path) -> bool:
    root = Path(path)
    return (root / "adapter_config.json").is_file() and any(
        (root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def latest_trainer_checkpoint(path: str | Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    root = Path(path)
    if not root.is_dir():
        return None
    for child in root.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            candidates.append((int(match.group(1)), child))
    return max(candidates, default=(0, None))[1]


def model_fingerprint_from_directory(model_dir: str | Path) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base model config.json is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    text = payload.get("text_config") or payload
    vision = payload.get("vision_config") or {}
    return {
        "model_type": payload.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "vocab_size": text.get("vocab_size"),
        "vision_hidden_size": vision.get("hidden_size"),
        "vision_depth": vision.get("depth"),
    }


def validate_r0_adapter_contract(
    adapter_dir: str | Path,
    model_dir: str | Path,
    *,
    expected_r: int = 16,
    expected_alpha: int = 32,
    expected_target_modules: Sequence[str],
) -> dict[str, Any]:
    """在 R1 启动前拒绝 2B、H1/H2、损坏或 LoRA 契约不匹配的 parent。"""

    root = Path(adapter_dir)
    if not adapter_is_complete(root):
        raise ValueError(f"R0 adapter is incomplete: {root}")
    manifest_path = root / "strategy_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"R0 adapter strategy_manifest.json is missing: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    if manifest.get("training_stage") != R0_STAGE:
        raise ValueError(f"R1 parent must be a formal {R0_STAGE} adapter")
    if manifest.get("strategy") != "lora":
        raise ValueError("R1 parent strategy must be lora")
    if manifest.get("checkpoint_type") not in {None, "adapter"}:
        raise ValueError("R1 parent must not contain prior visual specialization")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("R1 parent adapter_config.peft_type must be LORA")
    expected_targets = sorted(str(value) for value in expected_target_modules)
    actual_targets = sorted(str(value) for value in adapter_config.get("target_modules", []))
    lora_mismatches = {}
    if int(adapter_config.get("r", -1)) != int(expected_r):
        lora_mismatches["r"] = adapter_config.get("r")
    if int(adapter_config.get("lora_alpha", -1)) != int(expected_alpha):
        lora_mismatches["alpha"] = adapter_config.get("lora_alpha")
    if actual_targets != expected_targets:
        lora_mismatches["target_modules"] = actual_targets
    if lora_mismatches:
        raise ValueError(f"R0 adapter LoRA contract mismatch: {lora_mismatches}")

    current = model_fingerprint_from_directory(model_dir)
    parent = manifest.get("base_model_fingerprint")
    if not isinstance(parent, Mapping):
        raise ValueError("R0 adapter lacks base_model_fingerprint")
    fields = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "vision_hidden_size",
        "vision_depth",
    )
    mismatches = {
        field: {"model": current.get(field), "adapter": parent.get(field)}
        for field in fields
        if current.get(field) != parent.get(field)
    }
    if mismatches:
        raise ValueError(f"R0 adapter/base fingerprint mismatch: {mismatches}")
    weights = next(
        root / name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (root / name).is_file()
    )
    return {
        "valid": True,
        "adapter_dir": str(root),
        "adapter_sha256": file_sha256(weights),
        "training_stage": R0_STAGE,
        "base_fingerprint": current,
        "lora": {
            "r": expected_r,
            "alpha": expected_alpha,
            "target_modules": expected_targets,
        },
    }


def validate_stage2_sampler_coverage(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_batch_pattern: Sequence[str],
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """证明 coverage_first sampler 不截断 Stage2 尾部且不产生 source starvation。"""

    # Keep this contract tolerant of read_jsonl()'s lazy iterator while making
    # the rows reusable for sampler construction, length checks, and indexing.
    rows = list(rows)
    sampler = build_alternating_source_sampler(
        rows,
        source_batch_pattern,
        batch_size=batch_size,
        seed=seed,
        exhaustion_policy="coverage_first",
    )
    indices = list(iter(sampler))
    expected = set(range(len(rows)))
    actual = set(indices)
    duplicate_count = len(indices) - len(actual)
    missing = sorted(expected.difference(actual))
    sources_seen = {
        str(dict(rows[index].get("metadata", {})).get("training_source", "unknown"))
        for index in indices
    }
    required_sources = set(str(value) for value in source_batch_pattern)
    report = {
        "sample_count": len(rows),
        "sampler_exposures": len(indices),
        "unique_indices": len(actual),
        "duplicate_count": duplicate_count,
        "missing_indices": missing,
        "sources_seen": sorted(sources_seen),
        "source_starvation": sorted(required_sources.difference(sources_seen)),
        "valid": not missing
        and duplicate_count == 0
        and not required_sources.difference(sources_seen),
    }
    if not report["valid"]:
        raise ValueError(f"Stage2 coverage_first sampler validation failed: {report}")
    return report


def training_command(
    python_executable: str,
    *,
    config: str | Path,
    train_file: str | Path,
    validation_file: str | Path,
    output_dir: str | Path,
    save_steps: int,
    initial_adapter: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    mode: str | None = None,
    max_train_samples: int | None = None,
) -> list[str]:
    """构造共用 LoRA Trainer 命令，区分 adapter continuation 与 Trainer resume。"""

    command = [
        python_executable,
        "scripts/train_qwen3vl_lora.py",
        "--config",
        str(config),
        "--train-file",
        str(train_file),
        "--val-file",
        str(validation_file),
        "--output-dir",
        str(output_dir),
        "--save-steps",
        str(save_steps),
    ]
    if initial_adapter is not None:
        command.extend(["--initial-adapter", str(initial_adapter)])
    if resume_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_checkpoint)])
    if max_train_samples is not None:
        command.extend(["--max-train-samples", str(max_train_samples)])
    if mode is not None:
        command.append(mode)
    return command
