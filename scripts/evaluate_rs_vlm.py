"""Qwen3-VL + LoRA 遥感任务评测脚本。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.evaluation.checkpoint_loader import (
    load_finetuned_checkpoint,
    read_strategy_manifest,
)
from sat_rs_vlm.training.utils import (
    MODEL_DEPS_ERROR,
    model_input_device,
    move_to_device,
    resolve_torch_dtype,
    safe_import_model_dependencies,
)
from sat_rs_vlm.utils.jsonl import write_jsonl


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
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def extract_reference(messages: list[dict[str, Any]]) -> str:
    """从 messages 中提取 assistant 标准答案。"""

    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def extract_number(text: str) -> float | None:
    """从回答中抽取第一个数字。"""

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def keyword_hit(prediction: str, reference: str) -> bool:
    """计算简单关键词命中。"""

    tokens = set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", reference.lower()))
    if not tokens:
        return False
    return any(token in prediction.lower() for token in tokens)


def valid_json(text: str) -> bool:
    """判断文本是否是合法 JSON。"""

    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


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


def validate_local_adapter(adapter_source: str, *, local_files_only: bool) -> None:
    """验证本地 LoRA adapter，防止路径错误时被 PEFT 当作远程仓库名。"""

    path = Path(adapter_source)
    if not path.is_dir():
        if local_files_only:
            raise FileNotFoundError(f"Local LoRA adapter directory does not exist: {path}")
        return
    required = path / "adapter_config.json"
    weight_candidates = (path / "adapter_model.safetensors", path / "adapter_model.bin")
    if not required.is_file():
        raise FileNotFoundError(f"LoRA adapter is missing adapter_config.json: {path}")
    if not any(candidate.is_file() for candidate in weight_candidates):
        raise FileNotFoundError(f"LoRA adapter weights do not exist in: {path}")


def compatible_model_class(transformers: Any) -> Any:
    """选择 Qwen3-VL 专用类或兼容的多模态 AutoModel。"""

    for name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise ImportError("Transformers does not provide a Qwen3-VL compatible model class.")


def load_model(config: dict[str, Any], modules: dict[str, Any]) -> tuple[Any, Any]:
    """加载 base model、LoRA adapter 和 processor。"""

    transformers = modules["transformers"]
    peft = modules["peft"]
    torch = modules["torch"]
    model_cfg = dict(config["model"])
    local_files_only = bool(model_cfg.get("local_files_only", True))
    base_model = resolve_model_source(str(model_cfg["base_model"]))
    processor_id = resolve_model_source(str(model_cfg.get("processor_id", base_model)))
    adapter_source = resolve_model_source(str(model_cfg["adapter_path"]))
    validate_local_adapter(adapter_source, local_files_only=local_files_only)
    processor = transformers.AutoProcessor.from_pretrained(
        processor_id,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=local_files_only,
    )
    model_cls = compatible_model_class(transformers)
    model_kwargs: dict[str, Any] = {
        "device_map": model_cfg.get("device_map", "auto"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "local_files_only": local_files_only,
    }
    dtype = resolve_torch_dtype(torch, str(model_cfg.get("torch_dtype", "auto")))
    model_kwargs["dtype"] = dtype if dtype is not None else "auto"
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    model = model_cls.from_pretrained(
        base_model,
        **model_kwargs,
    )
    model = peft.PeftModel.from_pretrained(
        model,
        adapter_source,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, processor


def build_generation_kwargs(generation_cfg: dict[str, Any]) -> dict[str, Any]:
    """构造 generate 参数；贪心解码时不传无效的 temperature。"""

    do_sample = bool(generation_cfg.get("do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_cfg.get("max_new_tokens", 256)),
        "do_sample": do_sample,
        "num_beams": int(generation_cfg.get("num_beams", 1)),
    }
    if do_sample:
        kwargs["temperature"] = float(generation_cfg.get("temperature", 1.0))
        if "top_p" in generation_cfg:
            kwargs["top_p"] = float(generation_cfg["top_p"])
        if "top_k" in generation_cfg:
            kwargs["top_k"] = int(generation_cfg["top_k"])
    return kwargs


def generate_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_cfg: dict[str, Any],
    torch: Any,
) -> str:
    """对单条样本生成回答。"""

    batch = collator([sample])
    input_length = int(batch["input_ids"].shape[-1])
    input_device = model_input_device(model, torch)
    batch = move_to_device(batch, input_device, torch)
    with torch.inference_mode():
        output_ids = model.generate(**batch, **build_generation_kwargs(generation_cfg))
    generated_ids = output_ids[:, input_length:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(decoded[0]).strip() if decoded else ""


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """按任务聚合基础指标。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["task_type"])].append(row)

    empty_predictions = sum(not row["prediction"].strip() for row in predictions)
    summary: dict[str, Any] = {
        "overall": {
            "num_samples": len(predictions),
            "empty_predictions": empty_predictions,
            "empty_prediction_rate": empty_predictions / len(predictions) if predictions else 0.0,
            "inference_latency_ms": (
                sum(float(row.get("inference_latency_ms", 0.0)) for row in predictions)
                / len(predictions)
                if predictions
                else None
            ),
        },
        "by_task": {},
    }
    for task_type, rows in grouped.items():
        exact = sum(row["prediction"].strip() == row["reference"].strip() for row in rows)
        keyword = sum(keyword_hit(row["prediction"], row["reference"]) for row in rows)
        avg_len = sum(len(row["prediction"]) for row in rows) / max(len(rows), 1)
        metrics: dict[str, Any] = {
            "num_samples": len(rows),
            "exact_match": exact / len(rows),
            "keyword_hit_rate": keyword / len(rows),
            "average_generation_length": avg_len,
            "empty_prediction_rate": sum(not row["prediction"].strip() for row in rows) / len(rows),
        }
        if task_type == "detection":
            metrics["valid_json_rate"] = sum(valid_json(row["prediction"]) for row in rows) / len(
                rows
            )
        if task_type == "counting":
            errors = []
            for row in rows:
                pred_num = extract_number(row["prediction"])
                ref_num = extract_number(row["reference"])
                if pred_num is not None and ref_num is not None:
                    errors.append(abs(pred_num - ref_num))
            metrics["mae"] = sum(errors) / len(errors) if errors else None
        summary["by_task"][task_type] = metrics
    summary["todo"] = "TODO: add mAP, CIDEr, BLEU/ROUGE and task-specific RS metrics."
    return summary


def evaluate(config_path: Path, checkpoint: Path | None = None) -> None:
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
        started = time.perf_counter()
        prediction = generate_prediction(
            model,
            processor,
            collator,
            sample,
            generation_cfg,
            torch,
        )
        predictions.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "prediction": prediction,
                "reference": extract_reference(sample["messages"]),
                "metadata": sample.get("metadata", {}),
                "inference_latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(dataset):
            print(f"Evaluated {index}/{len(dataset)} samples")

    output_cfg = dict(config["output"])
    if checkpoint is None:
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
        evaluate(Path(args.config), checkpoint)
    except ImportError as exc:
        raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
