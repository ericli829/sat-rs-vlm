"""将项目内部 rs_*.jsonl 转换为 Qwen3-VL chat message JSONL。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.prompt_templates import strengthen_answer, strengthen_instruction
from sat_rs_vlm.data.task_protocol import counting_json
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_TASKS = {
    "detection",
    "counting",
    "scene_classification",
    "captioning",
    "vqa",
    "change_detection",
    "segmentation",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Convert RS JSONL to Qwen3-VL chat JSONL.")
    parser.add_argument(
        "--config",
        default="configs/data/remote_sensing_data.yaml",
        help="Path to data YAML config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as file:
        loaded = dict(yaml.safe_load(file) or {})
    return dict(expand_environment(loaded, environ=os.environ, allow_unresolved=True))


def convert_sample_to_qwen3vl(sample: dict[str, Any]) -> dict[str, Any]:
    """转换单条内部样本为 Qwen3-VL messages。

    参数：
        sample：内部 rs_*.jsonl 样本，包含 id/task_type/images/instruction/answer/metadata。

    返回值：
        dict[str, Any]：Qwen3-VL chat JSONL 行。
    """

    task_type = str(sample["task_type"])
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task_type: {task_type}")
    metadata = dict(sample.get("metadata", {}))
    answer = str(sample["answer"])
    if task_type == "counting":
        normalized_count = counting_json(answer)
        if normalized_count is None:
            metadata.update({"original_task_type": "counting", "counting_unresolved": True})
            task_type = "vqa"
        else:
            answer = normalized_count
    instruction = strengthen_instruction(task_type, str(sample["instruction"]))
    answer = strengthen_answer(task_type, answer)
    content = [
        {"type": "image", "image": str(image_path)} for image_path in list(sample.get("images", []))
    ]
    content.append({"type": "text", "text": instruction})
    return {
        "id": str(sample["id"]),
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
        ],
        "task_type": task_type,
        "metadata": metadata,
    }


def convert_file(input_file: Path, output_file: Path) -> int:
    """转换一个 JSONL 文件。"""

    rows = [convert_sample_to_qwen3vl(row) for row in read_jsonl(input_file)]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_file, rows)
    print(f"Wrote {len(rows)} Qwen3-VL samples to {output_file}")
    return len(rows)


def convert_all(config_path: Path) -> dict[str, Path]:
    """按配置转换 train/val/test 三个 split。"""

    root = Path.cwd()
    config = load_yaml(config_path)
    processed = dict(config.get("processed", {}))
    pairs = {
        "train": (
            root / str(processed.get("train_file", "data/processed/rs_train.jsonl")),
            root / str(processed.get("qwen_train_file", "data/processed/qwen3vl_train.jsonl")),
        ),
        "val": (
            root / str(processed.get("val_file", "data/processed/rs_val.jsonl")),
            root / str(processed.get("qwen_val_file", "data/processed/qwen3vl_val.jsonl")),
        ),
        "test": (
            root / str(processed.get("test_file", "data/processed/rs_test.jsonl")),
            root / str(processed.get("qwen_test_file", "data/processed/qwen3vl_test.jsonl")),
        ),
    }
    outputs: dict[str, Path] = {}
    for split, (input_file, output_file) in pairs.items():
        if not input_file.exists():
            raise FileNotFoundError(
                f"Input file does not exist: {input_file}. "
                "Run: python scripts/prepare_rs_instruction_data.py --config "
                "configs/data/remote_sensing_data.yaml"
            )
        convert_file(input_file, output_file)
        outputs[split] = output_file
    return outputs


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    convert_all(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
