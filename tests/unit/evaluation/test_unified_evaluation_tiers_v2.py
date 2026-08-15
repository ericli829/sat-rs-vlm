from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sat_rs_vlm.evaluation.tier_builder import (
    build_unified_evaluation_tiers,
    distribution,
)
from sat_rs_vlm.evaluation.tiers import LEGACY_TIER_FILES
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_HASHES = {
    "E1": "e513ad879cfe75496b2bd4f28f076e61977861b60695a52487ad28f93c3cee07",
    "E2": "20ee6da734545ba213947d44bb5bb1ee930563ac1571a61c385594ecb43d7a17",
    "E3": "e104aefcd3a524c041479e50af95453d7c3095063bdb3ff3f075b21d2daaf48f",
}


def _row(sample_id: str, dataset: str, task: str, subtype: str, image: str) -> dict:
    metadata: dict[str, object] = {"dataset": dataset, "source_task": subtype}
    answer = "answer"
    if task == "detection":
        area = {"small": 0.005, "medium": 0.05, "large": 0.2}[subtype]
        metadata["bbox_clipped"] = [0.0, 0.0, area, 1.0]
        answer = '{"label":"vehicle","bbox":[0,0,0.1,0.1]}'
    elif task == "counting":
        answer = subtype.replace("5-9", "7").replace("10+", "12")
    elif task == "vqa":
        metadata["qa_type"] = subtype
    elif task == "change_detection":
        metadata["changeflag"] = int(subtype)
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image}]},
            {"role": "assistant", "content": answer},
        ],
        "metadata": metadata,
    }


def _fixture(tmp_path: Path, output_name: str) -> Path:
    data_root = tmp_path / "datasets"
    rows: list[dict] = []
    tasks = {
        "captioning": ["default"],
        "detection": ["small", "medium", "large"],
        "counting": ["0", "1", "2", "3", "4", "5-9", "10+"],
        "scene_classification": ["scene"],
        "vqa": ["quantity", "position", "direction"],
    }
    index = 0
    for task, subtypes in tasks.items():
        for subtype in subtypes:
            for _ in range(8):
                image = f"VRSBench/images/vrs-{index}.png"
                path = data_root / image
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
                rows.append(_row(f"vrs-{index}", "VRSBench", task, subtype, image))
                index += 1
    for flag in ("0", "1"):
        for item in range(24):
            image = f"LEVIR-CC/images/{flag}-{item}.png"
            path = data_root / image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
            rows.append(
                _row(f"levir-{flag}-{item}", "LEVIR-CC", "change_detection", flag, image)
            )
    source = tmp_path / "population.jsonl"
    train = tmp_path / "train.jsonl"
    write_jsonl(source, rows)
    write_jsonl(train, [_row("train-only", "VRSBench", "vqa", "quantity", "unused.png")])
    config = {
        "schema_version": "2.0",
        "tier_version": "unified-v2",
        "seed": 42,
        "output_dir": str(tmp_path / output_name),
        "evaluation_scope": {
            "datasets": ["VRSBench", "LEVIR-CC"],
            "tasks": [*tasks, "change_detection"],
            "evaluation_unit": {
                "VRSBench": "annotation_task_case",
                "LEVIR-CC": "one_deterministic_reference_per_image_pair",
            },
        },
        "data": {
            "common_image_root": str(data_root),
            "source_files": [str(source)],
            "train_files": [str(train)],
        },
        "stratification": {
            "dataset_weights": {"VRSBench": 0.8, "LEVIR-CC": 0.2},
            "task_balance": "sqrt_population",
            "source_shortage_policy": "redistribute",
            "detection_area_thresholds": {"small_max": 0.01, "medium_max": 0.1},
        },
        "tiers": {
            "E1": {"target_samples": 30},
            "E2": {"target_samples": 90},
            "E3": {"mode": "full"},
        },
    }
    path = tmp_path / f"{output_name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_legacy_tier_canonical_hashes_are_unchanged() -> None:
    for tier, relative in LEGACY_TIER_FILES.items():
        payload = (PROJECT_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == LEGACY_HASHES[tier]


def test_unified_tiers_are_nested_multisource_deterministic_and_consistent(
    tmp_path: Path,
) -> None:
    first = build_unified_evaluation_tiers(_fixture(tmp_path, "first"), project_root=tmp_path)
    second = build_unified_evaluation_tiers(_fixture(tmp_path, "second"), project_root=tmp_path)

    first_rows = {
        tier: list(read_jsonl(Path(record["path"]))) for tier, record in first["tiers"].items()
    }
    ids = {tier: {row["id"] for row in rows} for tier, rows in first_rows.items()}
    assert ids["E1"] < ids["E2"] < ids["E3"]
    assert len(ids["E1"]) == 30
    assert len(ids["E2"]) == 90
    assert {row["metadata"]["dataset"] for row in first_rows["E1"]} == {
        "VRSBench",
        "LEVIR-CC",
    }
    assert {row["task_type"] for row in first_rows["E2"]} == {
        "captioning",
        "detection",
        "counting",
        "scene_classification",
        "vqa",
        "change_detection",
    }
    levir_flags = {
        row["metadata"]["changeflag"]
        for row in first_rows["E2"]
        if row["metadata"]["dataset"] == "LEVIR-CC"
    }
    assert levir_flags == {0, 1}
    assert first["train_evaluation_overlap"] == 0
    assert first["image_path_validation"]["valid"] is True
    assert first["tiers"]["E2"]["distribution"] == distribution(
        first_rows["E2"], small_max=0.01, medium_max=0.1
    )
    for tier in ("E1", "E2", "E3"):
        assert first["tiers"][tier]["sha256"] == second["tiers"][tier]["sha256"]


def test_unified_tier_builder_fails_with_sample_context_for_missing_image(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path, "broken")
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    source = Path(payload["data"]["source_files"][0])
    rows = list(read_jsonl(source))
    rows[0]["messages"][0]["content"][0]["image"] = "VRSBench/missing.png"
    write_jsonl(source, rows)

    with pytest.raises(FileNotFoundError, match=f"sample={rows[0]['id']}.*dataset=VRSBench"):
        build_unified_evaluation_tiers(config, project_root=tmp_path)
