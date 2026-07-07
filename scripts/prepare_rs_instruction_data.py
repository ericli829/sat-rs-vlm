"""准备统一遥感指令数据。

该脚本读取 configs/data/remote_sensing_data.yaml，将多来源遥感数据转换为项目内部
rs_*.jsonl 格式。第三阶段第一版保留真实数据集接入接口；当真实目录不存在时，自动
生成覆盖单图任务和双图变化检测任务的 sample 数据，便于训练 pipeline smoke test。
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import random
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

from sat_rs_vlm.utils.jsonl import write_jsonl

TASK_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    (
        "captioning",
        "请描述这张遥感图像中的主要地物。",
        "图像中包含建筑物、道路和植被区域。",
    ),
    ("vqa", "图像中是否存在道路？", "是，图像中可以观察到道路结构。"),
    ("detection", "请检测图像中的飞机。", "检测到疑似飞机目标，位于机场跑道附近。"),
    ("counting", "请统计图像中的建筑物数量。", "图像中大约有 5 个建筑物。"),
    ("scene_classification", "请判断这张遥感图像的场景类别。", "场景类别为城市建设区。"),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Prepare remote-sensing instruction JSONL data.")
    parser.add_argument(
        "--config",
        default="configs/data/remote_sensing_data.yaml",
        help="Path to data YAML config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """生成 PNG chunk。"""

    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_placeholder_png(path: Path, rgb: tuple[int, int, int]) -> None:
    """写入小尺寸 PNG 占位图。

    使用标准库手写 PNG，避免数据准备脚本依赖 pillow。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 32, 32
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(raw))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def ensure_sample_images(root: Path) -> dict[str, Path]:
    """确保 sample 图片存在。"""

    sample_dir = root / "data" / "samples"
    paths = {
        "demo": sample_dir / "demo_image.png",
        "before": sample_dir / "before.png",
        "after": sample_dir / "after.png",
    }
    colors = {"demo": (90, 130, 90), "before": (70, 100, 140), "after": (150, 120, 80)}
    for key, path in paths.items():
        if not path.exists():
            write_placeholder_png(path, colors[key])
    return paths


def real_dataset_dirs_exist(config: dict[str, Any], root: Path) -> bool:
    """检查是否存在任意真实数据集目录。"""

    raw_data = dict(config.get("raw_data", {}))
    return any((root / str(path)).exists() for path in raw_data.values())


def build_sample(split: str, index: int, image_path: Path, root: Path) -> dict[str, Any]:
    """构造单图 sample 样本。"""

    task_type, instruction, answer = TASK_TEMPLATES[index % len(TASK_TEMPLATES)]
    return {
        "id": f"{split}_{task_type}_{index:06d}",
        "task_type": task_type,
        "images": [image_path.relative_to(root).as_posix()],
        "instruction": instruction,
        "answer": answer,
        "metadata": {"dataset": "sample", "split": split},
    }


def build_change_sample(
    split: str,
    index: int,
    before_path: Path,
    after_path: Path,
    root: Path,
) -> dict[str, Any]:
    """构造双图变化检测 sample 样本。"""

    return {
        "id": f"{split}_change_{index:06d}",
        "task_type": "change_detection",
        "images": [
            before_path.relative_to(root).as_posix(),
            after_path.relative_to(root).as_posix(),
        ],
        "instruction": "第一张为变化前，第二张为变化后。请描述两张遥感图像之间的变化。",
        "answer": "变化后图像中新增了建筑物，道路区域基本保持不变。",
        "metadata": {"dataset": "sample_change", "split": split},
    }


def build_sample_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    """生成覆盖多任务的 sample 数据。"""

    images = ensure_sample_images(root)
    split_sizes = {"train": 20, "val": 8, "test": 4}
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, size in split_sizes.items():
        split_rows: list[dict[str, Any]] = []
        for index in range(size):
            if index % 6 == 5:
                split_rows.append(
                    build_change_sample(split, index, images["before"], images["after"], root)
                )
            else:
                split_rows.append(build_sample(split, index, images["demo"], root))
        rows[split] = split_rows
    return rows


def load_real_dataset_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """真实数据集接入预留接口。

    第一版暂不解析具体 benchmark 的原始标注格式。后续可在这里接入 VRSBench、
    MME Real RS、XLRS-bench、LEVIR-CC，并返回统一 rs_*.jsonl 行。
    """

    del config, root
    return []


def split_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """按配置比例切分真实数据行。"""

    split_cfg = dict(config.get("split", {}))
    rng = random.Random(int(split_cfg.get("seed", 42)))
    shuffled = list(rows)
    rng.shuffle(shuffled)
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_data(config_path: Path) -> dict[str, Path]:
    """执行数据准备并返回输出路径。"""

    root = Path.cwd()
    config = load_yaml(config_path)
    processed = dict(config.get("processed", {}))
    output_dir = root / str(processed.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if real_dataset_dirs_exist(config, root):
        real_rows = load_real_dataset_rows(config, root)
        rows_by_split = split_rows(real_rows, config) if real_rows else build_sample_rows(root)
    else:
        rows_by_split = build_sample_rows(root)

    outputs = {
        "train": root / str(processed.get("train_file", "data/processed/rs_train.jsonl")),
        "val": root / str(processed.get("val_file", "data/processed/rs_val.jsonl")),
        "test": root / str(processed.get("test_file", "data/processed/rs_test.jsonl")),
    }
    for split, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows_by_split[split])
        print(f"Wrote {len(rows_by_split[split])} {split} samples to {path}")
    return outputs


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    prepare_data(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
