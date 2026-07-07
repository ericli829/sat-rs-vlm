"""Qwen3-VL + LoRA 遥感任务评测脚本。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import sys
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
from sat_rs_vlm.training.utils import MODEL_DEPS_ERROR, safe_import_model_dependencies
from sat_rs_vlm.utils.jsonl import write_jsonl


def parse_args() -> argparse.Namespace:
    """解析评测参数。"""

    parser = argparse.ArgumentParser(description="Evaluate remote-sensing VLM predictions.")
    parser.add_argument(
        "--config",
        default="configs/eval/qwen3vl_eval.yaml",
        help="Path to eval YAML config.",
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


def load_model(config: dict[str, Any], modules: dict[str, Any]) -> tuple[Any, Any]:
    """加载 base model、LoRA adapter 和 processor。"""

    transformers = modules["transformers"]
    peft = modules["peft"]
    model_cfg = dict(config["model"])
    processor = transformers.AutoProcessor.from_pretrained(
        model_cfg["processor_id"],
        trust_remote_code=True,
    )
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = transformers.AutoModelForVision2Seq
    model = model_cls.from_pretrained(
        model_cfg["base_model"],
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model = peft.PeftModel.from_pretrained(model, model_cfg["adapter_path"])
    model.eval()
    return model, processor


def generate_prediction(
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    sample: dict[str, Any],
    generation_cfg: dict[str, Any],
) -> str:
    """对单条样本生成回答。"""

    batch = collator([sample])
    input_length = int(batch["input_ids"].shape[-1])
    output_ids = model.generate(
        **{key: value for key, value in batch.items() if key != "labels"},
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
        do_sample=bool(generation_cfg.get("do_sample", False)),
        temperature=float(generation_cfg.get("temperature", 0.0)),
        num_beams=int(generation_cfg.get("num_beams", 1)),
    )
    generated_ids = output_ids[:, input_length:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return str(decoded[0]).strip() if decoded else ""


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """按任务聚合基础指标。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["task_type"])].append(row)

    summary: dict[str, Any] = {"overall": {"num_samples": len(predictions)}, "by_task": {}}
    for task_type, rows in grouped.items():
        exact = sum(row["prediction"].strip() == row["reference"].strip() for row in rows)
        keyword = sum(keyword_hit(row["prediction"], row["reference"]) for row in rows)
        avg_len = sum(len(row["prediction"]) for row in rows) / max(len(rows), 1)
        metrics: dict[str, Any] = {
            "num_samples": len(rows),
            "exact_match": exact / len(rows),
            "keyword_hit_rate": keyword / len(rows),
            "average_generation_length": avg_len,
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


def evaluate(config_path: Path) -> None:
    """执行评测。"""

    config = load_yaml(config_path)
    modules = safe_import_model_dependencies(require_bitsandbytes=False)
    model, processor = load_model(config, modules)
    data_cfg = dict(config["data"])
    generation_cfg = dict(config.get("generation", {}))
    dataset = Qwen3VLDataset(data_cfg["eval_file"], data_cfg.get("max_eval_samples"))
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=4096,
        image_root=data_cfg["image_root"],
    )

    predictions: list[dict[str, Any]] = []
    for sample in dataset:
        prediction = generate_prediction(model, processor, collator, sample, generation_cfg)
        predictions.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "prediction": prediction,
                "reference": extract_reference(sample["messages"]),
                "metadata": sample.get("metadata", {}),
            }
        )

    output_cfg = dict(config["output"])
    summary_file = Path(output_cfg["summary_file"])
    predictions_file = Path(output_cfg["predictions_file"])
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
        evaluate(Path(args.config))
    except ImportError as exc:
        raise SystemExit(str(exc) or MODEL_DEPS_ERROR) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
