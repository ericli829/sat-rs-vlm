"""VRSBench 原始标注到项目内部指令格式的适配器。

VRSBench 每张图对应一个 JSON 标注，包含 caption、objects、qa_pairs 和 image。
本模块按图像级别流式读取标注，并展开为 captioning、detection、counting、
scene_classification 和 vqa 样本，避免一次性把完整数据集加载到内存。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VRSBenchLayout:
    """VRSBench 文件布局。

    参数：
        root：数据集根目录。
        train_images/val_images：相对 root 的图像目录。
        train_annotations/val_annotations：相对 root 的逐图 JSON 标注目录。
    """

    root: Path
    train_images: Path
    val_images: Path
    train_annotations: Path
    val_annotations: Path

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        project_root: Path,
    ) -> VRSBenchLayout:
        """从 YAML 字典解析并验证数据集布局。"""

        raw_root = Path(str(config.get("root", "data/raw/vrsbench"))).expanduser()
        root = raw_root if raw_root.is_absolute() else project_root / raw_root
        layout = cls(
            root=root,
            train_images=root / str(config.get("train_images", "Images/Images_train")),
            val_images=root / str(config.get("val_images", "Images/Images_val")),
            train_annotations=root
            / str(config.get("train_annotations", "Annotations/Annotations_train")),
            val_annotations=root
            / str(config.get("val_annotations", "Annotations/Annotations_val")),
        )
        layout.validate()
        return layout

    def validate(self) -> None:
        """检查根目录和四个必要 split 目录是否存在。"""

        paths = {
            "root": self.root,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "train_annotations": self.train_annotations,
            "val_annotations": self.val_annotations,
        }
        missing = [f"{name}={path}" for name, path in paths.items() if not path.is_dir()]
        if missing:
            raise FileNotFoundError("VRSBench directories do not exist: " + ", ".join(missing))

    def split_paths(self, split: str) -> tuple[Path, Path] | None:
        """返回 split 对应的图像和标注目录；VRSBench 没有独立 test 标注。"""

        if split == "train":
            return self.train_images, self.train_annotations
        if split == "val":
            return self.val_images, self.val_annotations
        if split == "test":
            return None
        raise ValueError(f"Unsupported VRSBench split: {split}")


def clip_unit(value: Any) -> float:
    """把有限数值裁剪到闭区间 `[0, 1]`。"""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"VRSBench coordinate must be finite, got: {value}")
    return min(1.0, max(0.0, number))


def clipped_bbox(raw_bbox: Any) -> tuple[list[float], list[float], bool]:
    """校验并裁剪 VRSBench 的 `[x_min, y_min, x_max, y_max]`。

    返回值：
        `(raw, clipped, changed)`；裁剪后会重新排序端点，保证最小值不大于最大值。
    """

    raw = [float(value) for value in list(raw_bbox or [])]
    if len(raw) != 4:
        raise ValueError(f"VRSBench obj_coord must contain 4 values, got: {raw}")
    clipped = [clip_unit(value) for value in raw]
    x_min, x_max = sorted((clipped[0], clipped[2]))
    y_min, y_max = sorted((clipped[1], clipped[3]))
    normalized = [x_min, y_min, x_max, y_max]
    return raw, normalized, raw != normalized


def qa_task_type(qa_type: str) -> str:
    """把 VRSBench QA 子类型映射到框架任务类型。"""

    normalized = qa_type.strip().lower()
    if normalized == "object quantity":
        return "counting"
    if normalized == "scene type":
        return "scene_classification"
    return "vqa"


def image_relative_path(layout: VRSBenchLayout, image_dir: Path, image_name: str) -> str:
    """返回相对于数据集 root 的 POSIX 图片路径，并验证文件存在。"""

    image_path = image_dir / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"VRSBench image does not exist: {image_path}")
    return image_path.relative_to(layout.root).as_posix()


def base_metadata(split: str, annotation_path: Path, image_name: str) -> dict[str, Any]:
    """构造所有展开样本共享的可追溯元数据。"""

    return {
        "dataset": "VRSBench",
        "split": split,
        "source_annotation": annotation_path.name,
        "source_image": image_name,
    }


def annotation_to_samples(
    row: Mapping[str, Any],
    *,
    split: str,
    annotation_path: Path,
    image_path: str,
    include_caption: bool = True,
    include_detection: bool = True,
    include_qa: bool = True,
) -> Iterator[dict[str, Any]]:
    """把单张图的 VRSBench 标注展开为内部指令样本。"""

    image_name = str(row.get("image", ""))
    stem = Path(image_name).stem
    metadata = base_metadata(split, annotation_path, image_name)

    caption = str(row.get("caption", "")).strip()
    if include_caption and caption:
        yield {
            "id": f"vrsbench_{split}_{stem}_caption",
            "task_type": "captioning",
            "images": [image_path],
            "instruction": "Describe this remote sensing image in detail.",
            "answer": caption,
            "metadata": {**metadata, "source_task": "caption"},
        }

    if include_detection:
        for index, obj in enumerate(list(row.get("objects", []))):
            if not isinstance(obj, Mapping):
                raise ValueError(f"Invalid VRSBench object in {annotation_path}: {obj}")
            raw_bbox, bbox, was_clipped = clipped_bbox(obj.get("obj_coord"))
            label = str(obj.get("obj_cls", "object")).strip() or "object"
            referring = str(obj.get("referring_sentence", "")).strip()
            obj_id = obj.get("obj_id", index)
            answer = json.dumps(
                {"label": label, "bbox": bbox},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield {
                "id": f"vrsbench_{split}_{stem}_det_{index:03d}",
                "task_type": "detection",
                "images": [image_path],
                "instruction": (
                    f'Locate the object described as: "{referring}". '
                    "Return its class and normalized bounding box."
                ),
                "answer": answer,
                "metadata": {
                    **metadata,
                    "source_task": "referring",
                    "object_id": obj_id,
                    "bbox_raw": raw_bbox,
                    "bbox_clipped": bbox,
                    "coordinate_clipped": was_clipped,
                },
            }

    if include_qa:
        for index, qa in enumerate(list(row.get("qa_pairs", []))):
            if not isinstance(qa, Mapping):
                raise ValueError(f"Invalid VRSBench QA pair in {annotation_path}: {qa}")
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            qa_type = str(qa.get("type", "vqa"))
            ques_id = qa.get("ques_id", index)
            if not question or not answer:
                raise ValueError(f"Empty VRSBench QA pair in {annotation_path}: {qa}")
            yield {
                "id": f"vrsbench_{split}_{stem}_qa_{index:03d}",
                "task_type": qa_task_type(qa_type),
                "images": [image_path],
                "instruction": question,
                "answer": answer,
                "metadata": {
                    **metadata,
                    "source_task": "vqa",
                    "qa_type": qa_type,
                    "question_id": ques_id,
                },
            }


def iter_vrsbench_samples(
    layout: VRSBenchLayout,
    split: str,
    *,
    max_images: int | None = None,
    include_caption: bool = True,
    include_detection: bool = True,
    include_qa: bool = True,
) -> Iterator[dict[str, Any]]:
    """按标注文件名顺序流式生成一个 VRSBench split 的指令样本。"""

    split_paths = layout.split_paths(split)
    if split_paths is None:
        return
    image_dir, annotation_dir = split_paths
    annotation_paths = sorted(
        path
        for path in annotation_dir.glob("*.json")
        if not path.name.startswith("._") and "__MACOSX" not in path.parts
    )
    if max_images is not None:
        annotation_paths = annotation_paths[:max_images]
    for annotation_path in annotation_paths:
        with annotation_path.open("r", encoding="utf-8") as file:
            row = json.load(file)
        if not isinstance(row, Mapping):
            raise ValueError(f"VRSBench annotation must be a JSON object: {annotation_path}")
        missing = {"caption", "objects", "qa_pairs", "image"} - set(row)
        if missing:
            raise ValueError(
                f"VRSBench annotation {annotation_path} missing fields: {sorted(missing)}"
            )
        image_name = str(row["image"])
        image_path = image_relative_path(layout, image_dir, image_name)
        yield from annotation_to_samples(
            row,
            split=split,
            annotation_path=annotation_path,
            image_path=image_path,
            include_caption=include_caption,
            include_detection=include_detection,
            include_qa=include_qa,
        )
