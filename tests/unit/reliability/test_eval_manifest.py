import json
from pathlib import Path

import pytest

from sat_rs_vlm.data.reliability_manifest import build_reliability_eval_manifest
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

TASKS = ("captioning", "vqa", "counting", "detection", "scene_classification")


def _row(sample_id: str, task: str, image: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "task_type": task,
        "images": [image],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"question-{task}"},
                ],
            },
            {"role": "assistant", "content": f"answer-{task}"},
        ],
        "metadata": {"source": "test"},
    }


def _dataset(root: Path, *, leak: bool = False) -> Path:
    (root / "images").mkdir(parents=True)
    validation = []
    for task in TASKS:
        image = f"images/{task}.ppm"
        (root / image).write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        validation.append(_row(f"validation-{task}", task, image))
    train = [_row("train-caption", "captioning", "images/captioning.ppm")]
    if leak:
        train[0]["id"] = "validation-captioning"
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "validation.jsonl", validation)
    write_jsonl(root / "test.jsonl", [])
    write_jsonl(root / "smoke.jsonl", [])
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "test",
        "dataset_version": "1",
        "root_format": "embedded",
        "image_path_type": "relative",
        "coordinate_format": "xyxy",
        "coordinate_range": [0, 1],
        "splits": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
            "smoke": "smoke.jsonl",
        },
    }
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_balanced_eval_manifest_uses_relative_images(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    output = tmp_path / "project_metadata/reliability/eval.jsonl"

    statistics = build_reliability_eval_manifest(
        tmp_path,
        manifest,
        source_split="validation",
        output_path=output,
        samples_per_task=1,
        seed=2026,
    )
    rows = list(read_jsonl(output))

    assert statistics["num_samples"] == 5
    assert {row["task_type"] for row in rows} == set(TASKS)
    assert all(row["source_split"] == "validation" for row in rows)
    assert all(not Path(row["images"][0]).is_absolute() for row in rows)
    assert output.with_suffix(".stats.json").is_file()


def test_eval_manifest_rejects_split_leakage(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path, leak=True)

    with pytest.raises(ValueError, match="split leakage"):
        build_reliability_eval_manifest(
            tmp_path,
            manifest,
            source_split="validation",
            output_path=tmp_path / "eval.jsonl",
            samples_per_task=1,
            seed=1,
        )
