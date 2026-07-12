from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.data.vrsbench import (
    VRSBenchLayout,
    clipped_bbox,
    iter_vrsbench_samples,
    qa_task_type,
)


def create_vrsbench_fixture(root: Path) -> VRSBenchLayout:
    """创建包含一个 train 和一个 val 样本的微型 VRSBench。"""

    for split in ("train", "val"):
        image_dir = root / "Images" / f"Images_{split}"
        annotation_dir = root / "Annotations" / f"Annotations_{split}"
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        image_name = f"{split}_001.png"
        (image_dir / image_name).write_bytes(b"png")
        annotation = {
            "caption": "A remote sensing image with one building.",
            "objects": [
                {
                    "obj_id": 7,
                    "referring_sentence": "The building near the right edge.",
                    "obj_cls": "building",
                    "obj_coord": [-0.1, 0.2, 1.2, 0.8],
                }
            ],
            "qa_pairs": [
                {
                    "ques_id": 1,
                    "question": "How many buildings are visible?",
                    "answer": "1",
                    "type": "object quantity",
                },
                {
                    "ques_id": 2,
                    "question": "Is this an urban scene?",
                    "answer": "Yes",
                    "type": "scene type",
                },
                {
                    "ques_id": 3,
                    "question": "What color is the roof?",
                    "answer": "Gray",
                    "type": "object color",
                },
            ],
            "image": image_name,
        }
        (annotation_dir / f"{split}_001.json").write_text(
            json.dumps(annotation),
            encoding="utf-8",
        )
    return VRSBenchLayout.from_config({"root": str(root)}, root.parent)


def test_clipped_bbox_limits_and_orders_coordinates() -> None:
    raw, clipped, changed = clipped_bbox([1.2, 0.8, -0.1, 0.2])

    assert raw == [1.2, 0.8, -0.1, 0.2]
    assert clipped == [0.0, 0.2, 1.0, 0.8]
    assert changed is True


def test_clipped_bbox_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="4 values"):
        clipped_bbox([0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        clipped_bbox([0.1, 0.2, float("nan"), 0.4])


def test_qa_task_mapping() -> None:
    assert qa_task_type("object quantity") == "counting"
    assert qa_task_type("scene type") == "scene_classification"
    assert qa_task_type("object color") == "vqa"


def test_iter_vrsbench_samples_expands_tasks_and_clips_bbox(tmp_path: Path) -> None:
    layout = create_vrsbench_fixture(tmp_path / "VRSBench")

    rows = list(iter_vrsbench_samples(layout, "train"))

    assert [row["task_type"] for row in rows] == [
        "captioning",
        "detection",
        "counting",
        "scene_classification",
        "vqa",
    ]
    detection = rows[1]
    answer = json.loads(detection["answer"])
    assert answer == {"label": "building", "bbox": [0.0, 0.2, 1.0, 0.8]}
    assert detection["metadata"]["bbox_raw"] == [-0.1, 0.2, 1.2, 0.8]
    assert detection["metadata"]["coordinate_clipped"] is True
    assert detection["images"] == ["Images/Images_train/train_001.png"]


def test_vrsbench_has_no_independent_test_rows(tmp_path: Path) -> None:
    layout = create_vrsbench_fixture(tmp_path / "VRSBench")

    assert list(iter_vrsbench_samples(layout, "test")) == []
