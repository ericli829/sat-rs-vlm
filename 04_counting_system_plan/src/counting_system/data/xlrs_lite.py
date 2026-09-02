"""XLRS-Bench-lite 计数样本读取。

数据逻辑根路径：/root/autodl-fs/datasets/xlrsbench-lite
下载脚本会把 UHR 图落到 autodl-tmp，并在 autodl-fs 建符号链接。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..image_ops import region_from_named
from ..paths import dataset_root, load_config
from ..runtime import ImageRef, Region
from ..target import TargetSpec, extract_target_from_question

COUNTING_HINTS = (
    "counting",
    "overall_counting",
    "regional_counting",
    "counting_with_complex_reasoning",
    "counting_with_changing_detection",
)

REGION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("TOP_LEFT", re.compile(r"\b(top[ -]?left|upper[ -]?left)\b", re.I)),
    ("TOP_RIGHT", re.compile(r"\b(top[ -]?right|upper[ -]?right)\b", re.I)),
    ("BOTTOM_LEFT", re.compile(r"\b(bottom[ -]?left|lower[ -]?left)\b", re.I)),
    ("BOTTOM_RIGHT", re.compile(r"\b(bottom[ -]?right|lower[ -]?right)\b", re.I)),
    ("TOP_CENTER", re.compile(r"\b(top[ -]?center|upper[ -]?middle|top[ -]?middle)\b", re.I)),
    ("CENTER_RIGHT", re.compile(r"\b(center[ -]?right|right[ -]?center|middle[ -]?right)\b", re.I)),
    ("CENTER_LEFT", re.compile(r"\b(center[ -]?left|left[ -]?center|middle[ -]?left)\b", re.I)),
    ("TOP", re.compile(r"\b(top|upper)\b", re.I)),
    ("BOTTOM", re.compile(r"\b(bottom|lower)\b", re.I)),
    ("LEFT", re.compile(r"\bleft\b", re.I)),
    ("RIGHT", re.compile(r"\bright\b", re.I)),
    ("CENTER", re.compile(r"\b(center|middle|central)\b", re.I)),
]

CHOICE_LETTER = re.compile(r"^\s*([A-Ea-e])\b")
INT_IN_TEXT = re.compile(r"-?\d+")


@dataclass(slots=True)
class XLRSCountSample:
    sample_id: str
    question: str
    options: list[str]
    answer_letter: str
    answer_value: int | None
    category: str
    l2_category: str
    image_path: Path
    target: TargetSpec
    region_name: str | None
    entire: bool

    def image_ref(self) -> ImageRef:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.image_path) as im:
            w, h = im.size
        return ImageRef(path=str(self.image_path), image_id=self.sample_id, width=w, height=h)

    def visual_input(self) -> ImageRef | Region:
        image = self.image_ref()
        if self.entire or not self.region_name:
            return image
        return region_from_named(image, self.region_name)


def is_counting_row(row: dict[str, Any]) -> bool:
    category = str(row.get("category") or "")
    l2 = str(row.get("l2-category") or row.get("l2_category") or "")
    task = str(row.get("task") or "")
    compact = " ".join((category, l2, task)).lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    if any(h in compact for h in COUNTING_HINTS):
        return True
    if category or l2:
        return False
    question = str(row.get("question") or "").lower()
    return "how many" in question or "number of" in question


def parse_region_name(question: str) -> str | None:
    for name, pattern in REGION_PATTERNS:
        if pattern.search(question):
            return name
    return None


def parse_answer_value(answer: str, options: list[str]) -> tuple[str, int | None]:
    letter = ""
    match = CHOICE_LETTER.match(str(answer or ""))
    if match:
        letter = match.group(1).upper()
    text = str(answer or "")
    if letter and options:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            text = options[idx]
    nums = INT_IN_TEXT.findall(text)
    value = int(nums[0]) if nums else None
    return letter, value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_xlrs_counting(
    root: str | Path | None = None,
    *,
    max_samples: int | None = None,
) -> list[XLRSCountSample]:
    base = Path(root) if root else dataset_root()
    manifest = base / "counting.jsonl"
    if not manifest.exists():
        alt = base / "samples.jsonl"
        manifest = alt if alt.exists() else manifest
    if not manifest.exists():
        raise FileNotFoundError(
            f"XLRS counting jsonl not found under {base}. Run scripts/download_assets.py"
        )
    samples: list[XLRSCountSample] = []
    for row in _iter_jsonl(manifest):
        if max_samples is not None and len(samples) >= max_samples:
            break
        sample = _row_to_sample(row, base)
        if sample is not None:
            samples.append(sample)
    return samples


def _row_to_sample(row: dict[str, Any], base: Path) -> XLRSCountSample | None:
    question = str(row.get("question") or "")
    options = list(row.get("options") or row.get("multi-choice options") or [])
    image_rel = row.get("image_path") or row.get("path") or row.get("image")
    if not image_rel:
        return None
    image_path = Path(str(image_rel))
    if not image_path.is_absolute():
        image_path = base / image_path
    if not image_path.exists():
        return None
    letter, value = parse_answer_value(str(row.get("answer") or ""), options)
    region_name = row.get("region_name") or parse_region_name(question)
    l2 = str(row.get("l2_category") or row.get("l2-category") or "")
    category = str(row.get("category") or "")
    is_regional = "regional" in f"{l2} {category}".lower()
    if region_name or is_regional:
        entire = False
    elif "entire" in row:
        entire = bool(row["entire"])
    else:
        entire = True
    target = extract_target_from_question(question)
    sample_id = str(row.get("sample_id") or row.get("index") or image_path.stem)
    return XLRSCountSample(
        sample_id=sample_id,
        question=question,
        options=[str(x) for x in options],
        answer_letter=letter,
        answer_value=value,
        category=str(row.get("category") or ""),
        l2_category=l2,
        image_path=image_path,
        target=target,
        region_name=region_name,
        entire=entire,
    )


def default_xlrs_root() -> Path:
    return dataset_root(load_config())
