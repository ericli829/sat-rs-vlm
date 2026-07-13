"""PluginContext 明确公开的模型、数据、Trainer 和报告服务。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.plugins.context import PluginContext, PublicService
from sat_rs_vlm.training.utils import (
    count_trainable_parameters,
    model_input_device,
    move_to_device,
    resolve_torch_dtype,
    safe_import_model_dependencies,
    torch_device_summary,
)


def _compatible_model_class(transformers: Any) -> Any:
    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise ImportError("Transformers does not provide a Qwen3-VL compatible model class")


def build_public_services(*, require_bitsandbytes: bool) -> Mapping[str, PublicService]:
    """创建白名单服务；重依赖只在外部 runner 调用本函数时导入。"""

    cache: dict[str, Any] = {}

    def loaded_modules() -> dict[str, Any]:
        if not cache:
            cache.update(safe_import_model_dependencies(require_bitsandbytes=require_bitsandbytes))
        return cache

    def runtime_modules() -> Mapping[str, Any]:
        return MappingProxyType(loaded_modules())

    def inspect_environment() -> dict[str, Any]:
        modules = loaded_modules()
        torch = modules["torch"]
        transformers = modules["transformers"]
        return {
            "torch": getattr(torch, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
            "peft": getattr(modules["peft"], "__version__", None),
            "device": torch_device_summary(torch),
        }

    def load_processor(context: PluginContext, config: Mapping[str, Any]) -> Any:
        transformers = loaded_modules()["transformers"]
        model_config = dict(config.get("model", {}))
        return transformers.AutoProcessor.from_pretrained(
            str(context.processor_dir),
            local_files_only=bool(model_config.get("local_files_only", True)),
            trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        )

    def load_base_model(
        context: PluginContext,
        config: Mapping[str, Any],
        strategy_kwargs: Mapping[str, Any],
    ) -> Any:
        modules = loaded_modules()
        torch = modules["torch"]
        transformers = modules["transformers"]
        model_config = dict(config.get("model", {}))
        kwargs: dict[str, Any] = {
            "local_files_only": bool(model_config.get("local_files_only", True)),
            "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
            "device_map": model_config.get("device_map", "auto"),
        }
        dtype = resolve_torch_dtype(torch, str(model_config.get("torch_dtype", "bfloat16")))
        if dtype is not None:
            kwargs["dtype"] = dtype
        if model_config.get("attn_implementation"):
            kwargs["attn_implementation"] = model_config["attn_implementation"]
        overlap = set(kwargs).intersection(strategy_kwargs)
        if overlap:
            raise ValueError(f"Plugin model kwargs override protected fields: {sorted(overlap)}")
        kwargs.update(strategy_kwargs)
        return _compatible_model_class(transformers).from_pretrained(
            str(context.model_dir),
            **kwargs,
        )

    def create_dataset(path: Path, max_samples: int | None) -> Qwen3VLDataset:
        return Qwen3VLDataset(path, max_samples)

    def create_collator(
        processor: Any,
        max_seq_length: int,
        image_root: Path,
    ) -> Qwen3VLDataCollator:
        return Qwen3VLDataCollator(
            processor,
            max_seq_length=max_seq_length,
            image_root=image_root,
            debug_shapes=True,
        )

    def forward_probe(model: Any, collator: Any, sample: Any) -> float | None:
        torch = loaded_modules()["torch"]
        batch = collator([sample])
        batch = move_to_device(batch, model_input_device(model, torch), torch)
        model.eval()
        with torch.inference_mode():
            output = model(**batch)
        loss = getattr(output, "loss", None)
        return float(loss.detach().cpu()) if loss is not None else None

    def create_trainer(
        *,
        model: Any,
        context: PluginContext,
        arguments: Mapping[str, Any],
        train_dataset: Any,
        eval_dataset: Any,
        collator: Any,
        optimizer_groups: list[dict[str, Any]] | None,
        callbacks: list[Any],
    ) -> Any:
        modules = loaded_modules()
        torch = modules["torch"]
        transformers = modules["transformers"]
        values = dict(arguments)
        values["output_dir"] = str(context.output_dir)
        values.setdefault("remove_unused_columns", False)
        values.setdefault("report_to", "none")
        values.setdefault("save_strategy", "steps")
        values.setdefault("eval_strategy", "steps" if eval_dataset is not None else "no")
        try:
            training_arguments = transformers.TrainingArguments(**values)
        except TypeError:
            values["evaluation_strategy"] = values.pop("eval_strategy")
            training_arguments = transformers.TrainingArguments(**values)
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_arguments,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": collator,
            "callbacks": callbacks,
        }
        if optimizer_groups is not None:
            clean_groups = [
                {key: value for key, value in group.items() if key != "name"}
                for group in optimizer_groups
            ]
            optimizer = torch.optim.AdamW(clean_groups)
            trainer_kwargs["optimizers"] = (optimizer, None)
        return transformers.Trainer(**trainer_kwargs)

    def write_json_report(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def resolve_path(value: str | Path, base: Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()

    def parameter_summary(model: Any) -> dict[str, Any]:
        trainable, total, ratio = count_trainable_parameters(model)
        return {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_ratio": ratio,
        }

    def match_module_suffixes(model: Any, targets: list[str]) -> list[str]:
        names = [name for name, _ in model.named_modules() if name]
        return sorted(
            name
            for name in names
            if name in targets or any(name.endswith(f".{target}") for target in targets)
        )

    def inspect_model_modules(model: Any) -> dict[str, Any]:
        names = [name for name, _ in model.named_modules() if name]
        return {
            "all": names,
            "vision": [
                name
                for name in names
                if any(token in name.lower() for token in ("visual", "vision", "image_tower"))
            ],
            "projector": [
                name
                for name in names
                if any(
                    token in name.lower() for token in ("projector", "vision_proj", "visual.merger")
                )
            ],
            "embeddings": [
                name
                for name in names
                if name.lower().endswith(("embed_tokens", "word_embeddings", ".wte"))
            ],
            "lm_head": [name for name in names if name == "lm_head" or name.endswith(".lm_head")],
        }

    def save_adapter(model: Any, processor: Any, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir), safe_serialization=True)
        processor.save_pretrained(str(output_dir / "processor"))

    def save_full_model(model: Any, processor: Any, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir), safe_serialization=True)
        processor.save_pretrained(str(output_dir / "processor"))

    def training_arguments_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
        training = dict(config.get("training", {}))
        supported = {
            "num_train_epochs",
            "max_steps",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "lr_scheduler_type",
            "logging_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
            "bf16",
            "fp16",
            "gradient_checkpointing",
            "max_grad_norm",
            "deepspeed",
            "fsdp",
        }
        return {
            key: value for key, value in training.items() if key in supported and value is not None
        }

    return MappingProxyType(
        {
            "runtime_modules": runtime_modules,
            "inspect_environment": inspect_environment,
            "load_base_model": load_base_model,
            "load_processor": load_processor,
            "create_dataset": create_dataset,
            "create_collator": create_collator,
            "create_trainer": create_trainer,
            "forward_probe": forward_probe,
            "write_json_report": write_json_report,
            "resolve_path": resolve_path,
            "parameter_summary": parameter_summary,
            "match_module_suffixes": match_module_suffixes,
            "inspect_model_modules": inspect_model_modules,
            "save_adapter": save_adapter,
            "save_full_model": save_full_model,
            "training_arguments_from_config": training_arguments_from_config,
        }
    )
