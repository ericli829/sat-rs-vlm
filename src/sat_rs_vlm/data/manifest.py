"""遥感数据集 manifest、分片读取与完整性校验。"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from sat_rs_vlm.utils.jsonl import read_jsonl

WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


class DatasetManifest(BaseModel):
    """训练数据根目录的可迁移描述。

    `splits` 中的路径相对 manifest 所在数据集根目录解析。样本中的图片路径
    也必须是相对路径，运行时由 `dataset_root / relative_path` 得到真实文件。
    """

    schema_version: str = "1.0"
    dataset_name: str
    dataset_version: str
    root_format: Literal["external", "embedded"]
    image_path_type: Literal["relative"]
    coordinate_format: Literal["xyxy"]
    coordinate_range: tuple[float, float]
    splits: dict[str, str]
    statistics: str | None = None
    classes: list[str] = Field(default_factory=list)

    @field_validator("coordinate_range")
    @classmethod
    def validate_coordinate_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        """保证坐标范围为严格递增的有限区间。"""

        if len(value) != 2 or value[0] >= value[1]:
            raise ValueError("coordinate_range must contain [minimum, maximum].")
        return value

    @field_validator("splits")
    @classmethod
    def validate_splits(cls, value: dict[str, str]) -> dict[str, str]:
        """要求四个标准分片都在 manifest 中声明。"""

        required = {"train", "validation", "test", "smoke"}
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"Missing dataset split(s): {', '.join(missing)}")
        for name, path in value.items():
            if _is_absolute_path(path):
                raise ValueError(f"Split path must be relative: {name}={path}")
        return value


class DatasetValidationReport(BaseModel):
    """数据校验的机器可读报告。"""

    valid: bool
    dataset_root: str
    manifest_path: str
    sample_counts: dict[str, int] = Field(default_factory=dict)
    image_count: int = 0
    task_distribution: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or bool(WINDOWS_ABSOLUTE.match(value))


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """读取并用 Pydantic 校验 manifest。"""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetManifest.model_validate(payload)


def resolve_split_path(
    dataset_root: str | Path,
    manifest: DatasetManifest,
    split: str,
) -> Path:
    """解析分片 JSONL，拒绝路径穿越和未声明分片。"""

    if split not in manifest.splits:
        choices = ", ".join(sorted(manifest.splits))
        raise KeyError(f"Unknown split '{split}'. Available splits: {choices}")
    root = Path(dataset_root).resolve()
    candidate = (root / PurePosixPath(manifest.splits[split])).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Split path escapes dataset root: {manifest.splits[split]}")
    return candidate


def load_manifest_split(
    dataset_root: str | Path,
    manifest: DatasetManifest,
    split: str,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """读取 manifest 指定分片，并可限制样本数。"""

    path = resolve_split_path(dataset_root, manifest, split)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")
    rows = list(read_jsonl(path))
    return rows if max_samples is None else rows[:max_samples]


def _image_paths(sample: dict[str, Any]) -> list[str]:
    images = sample.get("images", [])
    if isinstance(images, str):
        return [images]
    if isinstance(images, list):
        return [str(item) for item in images]
    image = sample.get("image")
    return [str(image)] if image else []


def _boxes(sample: dict[str, Any]) -> list[list[float]]:
    candidates = sample.get("boxes")
    if candidates is None and isinstance(sample.get("metadata"), dict):
        candidates = sample["metadata"].get("boxes")
    if candidates is None:
        return []
    if not isinstance(candidates, list):
        raise ValueError("boxes must be a list")
    return [list(box) for box in candidates]


def _validate_box(box: list[float], low: float, high: float) -> str | None:
    if len(box) != 4:
        return f"bbox must contain four values: {box}"
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return f"bbox contains a non-numeric value: {box}"
    if not (low <= x1 < x2 <= high and low <= y1 < y2 <= high):
        return f"bbox is outside [{low}, {high}] or is not xyxy ordered: {box}"
    return None


def validate_dataset(
    dataset_root: str | Path,
    *,
    manifest_name: str = "dataset_manifest.json",
    sample_images: int = 16,
    random_seed: int = 42,
    verify_images: bool = True,
) -> DatasetValidationReport:
    """检查 manifest、分片交叉、图片、任务字段和 bbox。

    返回报告而非在首个错误处退出，便于一次修复多项数据问题。
    """

    root = Path(dataset_root).resolve()
    manifest_path = root / manifest_name
    if (
        manifest_name == "dataset_manifest.json"
        and not manifest_path.is_file()
        and (root / "project_metadata/dataset_manifest.json").is_file()
    ):
        manifest_path = root / "project_metadata/dataset_manifest.json"
    report = DatasetValidationReport(
        valid=False,
        dataset_root=str(root),
        manifest_path=str(manifest_path),
    )
    if not root.is_dir():
        report.errors.append(f"Dataset root does not exist: {root}")
        return report
    try:
        manifest = load_dataset_manifest(manifest_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        report.errors.append(str(exc))
        return report

    split_ids: dict[str, set[str]] = {}
    all_images: set[Path] = set()
    task_counts: Counter[str] = Counter()
    for split in manifest.splits:
        try:
            rows = load_manifest_split(root, manifest, split)
        except (FileNotFoundError, ValueError) as exc:
            report.errors.append(str(exc))
            continue
        report.sample_counts[split] = len(rows)
        ids: set[str] = set()
        for index, row in enumerate(rows, start=1):
            sample_id = str(row.get("id", "")).strip()
            if not sample_id:
                report.errors.append(f"{split}:{index} is missing id")
            elif sample_id in ids:
                report.errors.append(f"{split} contains duplicate id: {sample_id}")
            ids.add(sample_id)
            task_type = str(row.get("task_type", "")).strip()
            if not task_type:
                report.errors.append(f"{split}:{index} is missing task_type")
            else:
                task_counts[task_type] += 1
            paths = _image_paths(row)
            if not paths:
                report.errors.append(f"{split}:{index} has no image path")
            for image_path in paths:
                if _is_absolute_path(image_path):
                    report.errors.append(f"{split}:{index} uses absolute image path: {image_path}")
                    continue
                resolved = (root / PurePosixPath(image_path)).resolve()
                if root != resolved and root not in resolved.parents:
                    report.errors.append(
                        f"{split}:{index} image escapes dataset root: {image_path}"
                    )
                    continue
                if not resolved.is_file():
                    report.errors.append(f"{split}:{index} image does not exist: {image_path}")
                all_images.add(resolved)
            for box in _boxes(row):
                box_error = _validate_box(box, *manifest.coordinate_range)
                if box_error:
                    report.errors.append(f"{split}:{index} {box_error}")
        split_ids[split] = ids

    split_names = list(split_ids)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = split_ids[left].intersection(split_ids[right])
            if overlap:
                report.errors.append(
                    f"Splits {left} and {right} overlap: {', '.join(sorted(overlap)[:5])}"
                )

    if verify_images:
        try:
            from PIL import Image
        except ImportError:
            report.warnings.append("Pillow is unavailable; image decoding was skipped.")
        else:
            candidates = sorted(path for path in all_images if path.is_file())
            random.Random(random_seed).shuffle(candidates)
            for decoded_path in candidates[: max(sample_images, 0)]:
                try:
                    with Image.open(decoded_path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    report.errors.append(f"Unreadable image {decoded_path}: {exc}")

    report.image_count = len(all_images)
    report.task_distribution = dict(sorted(task_counts.items()))
    report.valid = not report.errors
    return report
