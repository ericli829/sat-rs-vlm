"""Qwen3-VL + LoRA 遥感任务评测脚本。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from sat_rs_vlm.models.qwen3vl_loader import (
    load_qwen3vl,
)
from sat_rs_vlm.models.qwen3vl_loader import (
    validate_local_adapter as _validate_local_adapter,
)

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.evaluation.checkpoint_loader import (
    load_finetuned_checkpoint,
    read_strategy_manifest,
)
from sat_rs_vlm.evaluation.inference import (
    CHANGE_BINARY_PROMPT_VERSION,
    change_binary_inference_enabled,
    extract_reference,
    timed_change_binary_prediction,
    timed_prediction,
)
from sat_rs_vlm.evaluation.inference import (
    build_generation_kwargs as _build_generation_kwargs,
)
from sat_rs_vlm.evaluation.inference import (
    generate_prediction as _generate_prediction,
)
from sat_rs_vlm.evaluation.metrics import summarize_predictions
from sat_rs_vlm.training.utils import (
    MODEL_DEPS_ERROR,
    resolve_torch_dtype,
    safe_import_model_dependencies,
)
from sat_rs_vlm.utils.jsonl import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_generation_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容旧脚本导入，委托统一生成参数实现。"""

    return _build_generation_kwargs(generation_cfg)


def generate_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_cfg: dict[str, Any],
    torch: Any,
) -> str:
    """兼容旧脚本导入，委托统一新增 token 解码实现。"""

    return _generate_prediction(model, processor, collator, sample, generation_cfg, torch)


def validate_local_adapter(adapter_source: str, *, local_files_only: bool) -> None:
    """兼容旧评测测试和调用方。"""

    _validate_local_adapter(adapter_source, local_files_only=local_files_only)


def parse_args() -> argparse.Namespace:
    """解析评测参数。"""

    parser = argparse.ArgumentParser(description="Evaluate remote-sensing VLM predictions.")
    parser.add_argument(
        "--config",
        default="configs/eval/qwen3vl_eval.yaml",
        help="Path to eval YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Experiment directory containing strategy_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory; keeps the evaluated checkpoint read-only.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as file:
        loaded = dict(yaml.safe_load(file) or {})
    return dict(expand_environment(loaded, environ=os.environ, allow_unresolved=False))


def resolve_project_path(value: str | Path) -> Path:
    """把相对路径解析到项目根目录。"""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_model_source(value: str) -> str:
    """优先把存在的相对路径解析为本地路径，否则保留 HuggingFace 模型 ID。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    project_path = PROJECT_ROOT / path
    return str(project_path) if project_path.exists() else value


def load_model(config: dict[str, Any], modules: dict[str, Any]) -> tuple[Any, Any]:
    """加载 base model、LoRA adapter 和 processor。"""

    torch = modules["torch"]
    model_cfg = dict(config["model"])
    local_files_only = bool(model_cfg.get("local_files_only", True))
    base_model = resolve_model_source(str(model_cfg["base_model"]))
    processor_id = resolve_model_source(str(model_cfg.get("processor_id", base_model)))
    raw_adapter = model_cfg.get("adapter_path")
    adapter_source = resolve_model_source(str(raw_adapter)) if raw_adapter else None
    model_kwargs: dict[str, Any] = {
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "local_files_only": local_files_only,
    }
    dtype = resolve_torch_dtype(torch, str(model_cfg.get("torch_dtype", "auto")))
    model_kwargs["dtype"] = dtype if dtype is not None else "auto"
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    return load_qwen3vl(
        modules=modules,
        base_model=base_model,
        processor_source=processor_id,
        model_kwargs=model_kwargs,
        processor_kwargs={
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
            "local_files_only": local_files_only,
        },
        adapter_path=adapter_source,
    )


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """兼容旧调用，内部使用统一任务指标。"""

    return summarize_predictions(predictions)


def evaluate(
    config_path: Path,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """执行评测。"""

    config = load_yaml(config_path)
    data_cfg = dict(config["data"])
    eval_file = resolve_project_path(str(data_cfg["eval_file"]))
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation JSONL file does not exist: {eval_file}")
    require_bitsandbytes = False
    if checkpoint is not None:
        require_bitsandbytes = bool(read_strategy_manifest(checkpoint).get("quantized_base", False))
    modules = safe_import_model_dependencies(require_bitsandbytes=require_bitsandbytes)
    if checkpoint is None:
        model, processor = load_model(config, modules)
    else:
        model, processor, _ = load_finetuned_checkpoint(
            checkpoint,
            dict(config.get("model", {})),
            modules,
        )
    torch = modules["torch"]
    generation_cfg = dict(config.get("generation", {}))
    dataset = Qwen3VLDataset(eval_file, data_cfg.get("max_eval_samples"))
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=int(data_cfg.get("max_seq_length", 4096)),
        image_root=resolve_project_path(str(data_cfg["image_root"])),
        for_generation=True,
    )

    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset, start=1):
        prediction, latency_ms = timed_prediction(
            model,
            processor,
            collator,
            sample,
            generation_cfg,
            torch,
        )
        row: dict[str, Any] = {
            "id": sample["id"],
            "task_type": sample["task_type"],
            "prediction": prediction,
            "reference": extract_reference(sample["messages"]),
            "metadata": sample.get("metadata", {}),
            "inference_latency_ms": latency_ms,
        }
        if change_binary_inference_enabled(sample, generation_cfg):
            binary_raw, binary_flag, binary_latency_ms = timed_change_binary_prediction(
                model,
                processor,
                collator,
                sample,
                generation_cfg,
                torch,
            )
            row.update(
                {
                    "prediction_changeflag": binary_flag,
                    "binary_prediction": binary_raw,
                    "binary_prediction_parse_ok": binary_flag is not None,
                    "binary_prompt_version": CHANGE_BINARY_PROMPT_VERSION,
                    "binary_inference_latency_ms": binary_latency_ms,
                    "total_inference_latency_ms": latency_ms + binary_latency_ms,
                }
            )
        predictions.append(row)
        if index == 1 or index % 10 == 0 or index == len(dataset):
            print(f"Evaluated {index}/{len(dataset)} samples")

    output_cfg = dict(config["output"])
    if output_dir is not None:
        summary_file = output_dir.resolve() / "summary.json"
        predictions_file = output_dir.resolve() / "predictions.jsonl"
    elif checkpoint is None:
        summary_file = resolve_project_path(str(output_cfg["summary_file"]))
        predictions_file = resolve_project_path(str(output_cfg["predictions_file"]))
    else:
        eval_output = checkpoint.resolve() / "evaluation"
        summary_file = eval_output / "summary.json"
        predictions_file = eval_output / "predictions.jsonl"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summarize(predictions), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(predictions_file, predictions)
    print(f"Saved summary to {summary_file}")
    print(f"Saved predictions to {predictions_file}")


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        evaluate(Path(args.config), checkpoint, output_dir)
    except ImportError as exc:
        raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
