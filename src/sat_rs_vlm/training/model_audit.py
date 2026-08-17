"""Qwen3-VL LoRA target 与 adapter/base architecture 兼容性审计。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def model_fingerprint(model: Any) -> dict[str, Any]:
    """提取区分 2B/4B 的稳定结构字段，不依赖具体 Transformers 配置类。"""

    get_base_model = getattr(model, "get_base_model", None)
    fingerprint_model = get_base_model() if callable(get_base_model) else model
    config = getattr(fingerprint_model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    vision_config = getattr(config, "vision_config", None)
    payload = {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", None) or []),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "num_hidden_layers": getattr(text_config, "num_hidden_layers", None),
        "num_attention_heads": getattr(text_config, "num_attention_heads", None),
        "vocab_size": getattr(text_config, "vocab_size", None),
        "vision_hidden_size": getattr(vision_config, "hidden_size", None),
        "vision_depth": getattr(vision_config, "depth", None),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {**payload, "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def audit_lora_targets(model: Any, targets: Sequence[str]) -> dict[str, Any]:
    """逐 target 统计真实模块命中；任一 target 为零时 fail-fast。"""

    names = [name for name, _ in model.named_modules()]
    counts = {
        str(target): sum(
            name == str(target) or name.endswith(f".{target}") for name in names
        )
        for target in targets
    }
    missing = sorted(target for target, count in counts.items() if count == 0)
    if missing:
        raise ValueError(
            "LoRA target modules did not match the loaded model: " + ", ".join(missing)
        )
    return {
        "target_match_counts": counts,
        "matched_module_count": sum(counts.values()),
        "matched_modules": sorted(
            name
            for name in names
            if any(name == target or name.endswith(f".{target}") for target in counts)
        ),
    }


def finalize_lora_trainable_audit(
    model: Any, target_audit: Mapping[str, Any]
) -> dict[str, Any]:
    """在 PEFT 注入后补充每个 target 的可训练参数量和全模型占比。"""

    targets = list(dict(target_audit["target_match_counts"]))
    trainable_by_target = {target: 0 for target in targets}
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if not bool(parameter.requires_grad):
            continue
        trainable += count
        if "lora_" in name:
            for target in targets:
                if f".{target}." in name or name.startswith(f"{target}."):
                    trainable_by_target[target] += count
                    break
    return {
        **dict(target_audit),
        "trainable_parameters_by_target": trainable_by_target,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / max(1, total),
    }


def _load_adapter_fingerprint(adapter_dir: Path) -> Mapping[str, Any] | None:
    manifest_path = adapter_dir / "strategy_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest.get("base_model_fingerprint")
        if isinstance(fingerprint, Mapping):
            return fingerprint
    return None


def validate_adapter_architecture(
    model: Any,
    adapter_dir: str | Path,
    *,
    require_fingerprint: bool,
) -> dict[str, Any]:
    """比较 adapter manifest 与已加载 base；严格 cycle 不接受无指纹 adapter。"""

    path = Path(adapter_dir)
    current = model_fingerprint(model)
    adapter = _load_adapter_fingerprint(path)
    if adapter is None:
        if require_fingerprint:
            raise ValueError(
                "Initial adapter lacks base_model_fingerprint in strategy_manifest.json: "
                f"{path}. Refusing an unverified 2B/4B adapter chain."
            )
        return {"verified": False, "reason": "adapter_fingerprint_unavailable", "model": current}
    comparable = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    mismatches = {
        key: {"model": current.get(key), "adapter": adapter.get(key)}
        for key in comparable
        if current.get(key) != adapter.get(key)
    }
    if mismatches:
        raise ValueError(
            f"Initial adapter is incompatible with the loaded base model: {mismatches}"
        )
    return {"verified": True, "model": current, "adapter": dict(adapter)}
