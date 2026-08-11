"""依据 strategy_manifest 加载 Adapter 或完整模型 checkpoint。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class
from sat_rs_vlm.training.utils import resolve_torch_dtype
from sat_rs_vlm.training.vision_tuning import load_visual_sidecar


def read_strategy_manifest(checkpoint: str | Path) -> dict[str, Any]:
    """读取并验证 checkpoint 自描述文件。"""

    checkpoint_path = Path(checkpoint)
    manifest_path = checkpoint_path / "strategy_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing strategy_manifest.json: {checkpoint_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid strategy manifest object: {manifest_path}")
    if not str(payload.get("strategy", "")):
        raise ValueError(f"Strategy manifest has no strategy identifier: {manifest_path}")
    if not isinstance(payload.get("adapter_based"), bool):
        raise ValueError(f"Strategy manifest requires boolean adapter_based: {manifest_path}")
    return payload


def validate_checkpoint_files(checkpoint: str | Path, manifest: dict[str, Any]) -> None:
    """检查 manifest 所声明 checkpoint 类型的关键文件。"""

    checkpoint_path = Path(checkpoint)
    if bool(manifest["adapter_based"]):
        if not (checkpoint_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter config is missing: {checkpoint_path}")
        adapter_weight_files = (
            checkpoint_path / "adapter_model.safetensors",
            checkpoint_path / "adapter_model.bin",
        )
        if not any(path.is_file() for path in adapter_weight_files):
            raise FileNotFoundError(f"Adapter weights are missing: {checkpoint_path}")
        if manifest.get("checkpoint_type") == "adapter_with_visual_sidecar":
            sidecar_name = str(manifest.get("visual_sidecar", ""))
            if not sidecar_name:
                raise ValueError("H1 checkpoint manifest does not declare visual_sidecar")
            if not (checkpoint_path / sidecar_name).is_file():
                raise FileNotFoundError(
                    f"H1 visual sidecar is missing: {checkpoint_path / sidecar_name}"
                )
    else:
        if not (checkpoint_path / "config.json").is_file():
            raise FileNotFoundError(f"Full-model config.json is missing: {checkpoint_path}")
        full_weight_files = list(checkpoint_path.glob("*.safetensors")) + list(
            checkpoint_path.glob("pytorch_model*.bin")
        )
        if not full_weight_files:
            raise FileNotFoundError(f"Full-model weights are missing: {checkpoint_path}")
    processor_path = checkpoint_path / "processor"
    if not processor_path.is_dir():
        raise FileNotFoundError(f"Checkpoint processor directory is missing: {processor_path}")


def _base_model_kwargs(
    manifest: dict[str, Any],
    eval_model_config: dict[str, Any],
    modules: dict[str, Any],
) -> dict[str, Any]:
    """恢复普通精度或 QLoRA 量化加载参数。"""

    torch = modules["torch"]
    transformers = modules["transformers"]
    kwargs: dict[str, Any] = {
        "local_files_only": bool(eval_model_config.get("local_files_only", True)),
        "trust_remote_code": bool(eval_model_config.get("trust_remote_code", True)),
        "device_map": eval_model_config.get("device_map", "auto"),
    }
    dtype_name = str(eval_model_config.get("torch_dtype", manifest.get("actual_dtype", "auto")))
    dtype = resolve_torch_dtype(torch, dtype_name)
    kwargs["dtype"] = dtype if dtype is not None else "auto"
    if eval_model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = eval_model_config["attn_implementation"]
    if bool(manifest.get("quantized_base", False)):
        quant = manifest.get("quantization")
        if not isinstance(quant, dict) or not bool(quant.get("load_in_4bit")):
            raise ValueError("Quantized checkpoint manifest lacks its 4-bit configuration")
        compute_dtype = resolve_torch_dtype(
            torch,
            str(quant.get("compute_dtype", "bfloat16")),
        )
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant.get("use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return kwargs


def load_finetuned_checkpoint(
    checkpoint: str | Path,
    eval_model_config: dict[str, Any],
    modules: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """按 manifest 加载微调模型、Processor 和 manifest。"""

    checkpoint_path = Path(checkpoint).resolve()
    manifest = read_strategy_manifest(checkpoint_path)
    validate_checkpoint_files(checkpoint_path, manifest)
    transformers = modules["transformers"]
    peft = modules["peft"]
    local_files_only = bool(eval_model_config.get("local_files_only", True))
    processor = transformers.AutoProcessor.from_pretrained(
        str(checkpoint_path / "processor"),
        local_files_only=local_files_only,
        trust_remote_code=bool(eval_model_config.get("trust_remote_code", True)),
    )
    model_class = compatible_model_class(transformers)
    if bool(manifest["adapter_based"]):
        model_dir = str(manifest.get("model_dir", ""))
        if not model_dir:
            raise ValueError("Adapter manifest does not contain model_dir")
        model = model_class.from_pretrained(
            model_dir,
            **_base_model_kwargs(manifest, eval_model_config, modules),
        )
        model = peft.PeftModel.from_pretrained(
            model,
            str(checkpoint_path),
            local_files_only=local_files_only,
        )
        if manifest.get("checkpoint_type") == "adapter_with_visual_sidecar":
            load_visual_sidecar(model, checkpoint_path / str(manifest["visual_sidecar"]))
    else:
        model = model_class.from_pretrained(
            str(checkpoint_path),
            **_base_model_kwargs(manifest, eval_model_config, modules),
        )
    model.eval()
    return model, processor, manifest
