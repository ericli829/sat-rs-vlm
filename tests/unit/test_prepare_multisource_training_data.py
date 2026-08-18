from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/data/prepare_multisource_training_data.py"


def load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_multisource_training_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def internal_row(
    sample_id: str,
    images: list[str],
    *,
    task_type: str = "change_detection",
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "task_type": task_type,
        "images": images,
        "instruction": "Describe the changes.",
        "answer": "A building was added.",
        "metadata": {"dataset": "test"},
    }


def test_preparer_repairs_stale_windows_paths_and_groups_validation(tmp_path: Path) -> None:
    module = load_script_module()
    common_root = tmp_path / "datasets"
    source_root = common_root / "LEVIR-CC"
    for split in ("train", "val"):
        for period in ("A", "B"):
            directory = source_root / "images" / split / period
            directory.mkdir(parents=True)
            (directory / f"{split}_000001.png").write_bytes(b"png")

    stale_train = [
        rf"D:\old\Levir-CC-dataset\images\train\{period}\train_000001.png" for period in ("A", "B")
    ]
    stale_val = [
        rf"D:\old\Levir-CC-dataset\images\val\{period}\val_000001.png" for period in ("A", "B")
    ]
    train_file = source_root / "train.jsonl"
    val_file = source_root / "val.jsonl"
    write_jsonl(train_file, [internal_row("train-1", stale_train)])
    write_jsonl(
        val_file,
        [
            internal_row("val-caption-1", stale_val),
            internal_row("val-caption-2", stale_val),
        ],
    )
    train_output = tmp_path / "out/train.jsonl"
    val_output = tmp_path / "out/val.jsonl"
    report_output = tmp_path / "out/report.json"
    config = {
        "common_image_root": str(common_root),
        "seed": 42,
        "output": {
            "train_file": str(train_output),
            "validation_file": str(val_output),
            "report_file": str(report_output),
        },
        "sources": [
            {
                "name": "LEVIR-CC",
                "image_root": str(source_root),
                "train_file": str(train_file),
                "validation_file": str(val_file),
                "validation_samples": 10,
                "validation_group_by_images": True,
            }
        ],
    }

    report = module.prepare_multisource_data(config)
    train_row = next(read_jsonl(train_output))
    image_paths = [
        item["image"] for item in train_row["messages"][0]["content"] if item["type"] == "image"
    ]

    assert image_paths == [
        "LEVIR-CC/images/train/A/train_000001.png",
        "LEVIR-CC/images/train/B/train_000001.png",
    ]
    assert report["train_samples"] == 1
    assert report["validation_samples"] == 1
    assert report["sources"]["LEVIR-CC"]["validation_samples_available"] == 2
    assert json.loads(report_output.read_text(encoding="utf-8"))["valid"] is True


def test_preparer_rejects_change_sample_without_two_images(tmp_path: Path) -> None:
    module = load_script_module()
    common_root = tmp_path / "datasets"
    source_root = common_root / "LEVIR-CC"
    image = source_root / "images/train/A/one.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    train_file = source_root / "train.jsonl"
    val_file = source_root / "val.jsonl"
    row = internal_row("bad", ["images/train/A/one.png"])
    write_jsonl(train_file, [row])
    write_jsonl(val_file, [row | {"id": "bad-val"}])
    config = {
        "common_image_root": str(common_root),
        "output": {
            "train_file": str(tmp_path / "train-out.jsonl"),
            "validation_file": str(tmp_path / "val-out.jsonl"),
            "report_file": str(tmp_path / "report.json"),
        },
        "sources": [
            {
                "name": "LEVIR-CC",
                "image_root": str(source_root),
                "train_file": str(train_file),
                "validation_file": str(val_file),
            }
        ],
    }

    with pytest.raises(ValueError, match="exactly two images"):
        module.prepare_multisource_data(config)


def test_preparer_rotates_caption_variants_between_rounds(tmp_path: Path) -> None:
    module = load_script_module()
    common_root = tmp_path / "datasets"
    source_root = common_root / "LEVIR-CC"
    images = []
    for period in ("A", "B"):
        image = source_root / "images/train" / period / "pair.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        images.append(f"images/train/{period}/pair.png")
    val_images = []
    for period in ("A", "B"):
        image = source_root / "images/val" / period / "pair.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        val_images.append(f"images/val/{period}/pair.png")

    train_file = source_root / "train.jsonl"
    val_file = source_root / "val.jsonl"
    write_jsonl(
        train_file,
        [internal_row(f"caption-{index}", images) for index in range(5)],
    )
    write_jsonl(val_file, [internal_row("val-0", val_images)])
    config = {
        "common_image_root": str(common_root),
        "output": {
            "train_file": str(tmp_path / "round.jsonl"),
            "validation_file": str(tmp_path / "val-out.jsonl"),
            "report_file": str(tmp_path / "report.json"),
        },
        "sources": [
            {
                "name": "LEVIR-CC",
                "image_root": str(source_root),
                "train_file": str(train_file),
                "validation_file": str(val_file),
                "training_samples_per_image_group": 2,
            }
        ],
    }

    module.prepare_multisource_data(config, round_index=0)
    round_zero_ids = {row["id"] for row in read_jsonl(tmp_path / "round.jsonl")}
    module.prepare_multisource_data(config, round_index=1)
    round_one_ids = {row["id"] for row in read_jsonl(tmp_path / "round.jsonl")}

    assert round_zero_ids == {"caption-0", "caption-1"}
    assert round_one_ids == {"caption-2", "caption-3"}


def test_task_quotas_are_group_balanced() -> None:
    module = load_script_module()
    rows = []
    for image_index in range(3):
        for sample_index in range(3):
            rows.append(
                {
                    "id": f"d-{image_index}-{sample_index}",
                    "task_type": "detection",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": f"image-{image_index}.png"}],
                        }
                    ],
                }
            )

    selected = module._sample_task_quotas(
        rows,
        {"detection": 3},
        seed=42,
        group_by_images=True,
    )

    assert len({tuple(module._message_images(row)) for row in selected}) == 3


def test_full_evaluation_population_uses_one_case_per_image_pair(tmp_path: Path) -> None:
    module = load_script_module()
    common_root = tmp_path / "datasets"
    source_root = common_root / "LEVIR-CC"
    train_images: list[str] = []
    val_images: list[str] = []
    for split, destination in (("train", train_images), ("val", val_images)):
        for period in ("A", "B"):
            image = source_root / "images" / split / period / "pair.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"png")
            destination.append(f"images/{split}/{period}/pair.png")
    train_file = source_root / "train.jsonl"
    val_file = source_root / "val.jsonl"
    write_jsonl(train_file, [internal_row("train", train_images)])
    write_jsonl(
        val_file,
        [internal_row(f"val-caption-{index}", val_images) for index in range(5)],
    )
    output = tmp_path / "full-eval.jsonl"
    report_path = tmp_path / "full-eval-report.json"
    config = {
        "common_image_root": str(common_root),
        "seed": 42,
        "sources": [
            {
                "name": "LEVIR-CC",
                "image_root": str(source_root),
                "train_file": str(train_file),
                "validation_file": str(val_file),
                "validation_samples": 1,
                "validation_group_by_images": True,
            }
        ],
    }

    report = module.prepare_full_evaluation_population(
        config,
        evaluation_output=output,
        report_output=report_path,
    )

    assert len(list(read_jsonl(output))) == 1
    assert report["sources"]["LEVIR-CC"] == {
        "validation_rows_available": 5,
        "unique_image_groups": 1,
        "evaluation_cases_selected": 1,
        "evaluation_unit": "image_pair",
        "group_by_images": True,
        "reference_selection": "one_deterministic_reference_per_image_group",
        "task_distribution": {"change_detection": 1},
    }


def test_full_cycle_manifest_proves_coverage_and_is_reproducible(tmp_path: Path) -> None:
    module = load_script_module()
    common_root = tmp_path / "datasets"
    vrs_root = common_root / "VRSBench"
    levir_root = common_root / "LEVIR-CC"
    vrs_image = vrs_root / "images/vrs.png"
    vrs_image.parent.mkdir(parents=True)
    vrs_image.write_bytes(b"png")
    pair_images: list[str] = []
    for period in ("A", "B"):
        image = levir_root / "images/train" / period / "pair.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        pair_images.append(f"images/train/{period}/pair.png")
    val_pair_images: list[str] = []
    for period in ("A", "B"):
        image = levir_root / "images/val" / period / "pair.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        val_pair_images.append(f"images/val/{period}/pair.png")

    vrs_train = vrs_root / "train.jsonl"
    vrs_val = vrs_root / "val.jsonl"
    tasks = ["detection", "detection", "detection", "vqa", "counting"]
    write_jsonl(
        vrs_train,
        [
            internal_row(f"vrs-{index}", ["images/vrs.png"], task_type=task)
            for index, task in enumerate(tasks)
        ],
    )
    write_jsonl(vrs_val, [internal_row("vrs-val", ["images/vrs.png"], task_type="vqa")])
    levir_train = levir_root / "train.jsonl"
    levir_val = levir_root / "val.jsonl"
    write_jsonl(
        levir_train,
        [internal_row(f"levir-{index}", pair_images) for index in range(5)],
    )
    write_jsonl(levir_val, [internal_row("levir-val", val_pair_images)])
    protected = tmp_path / "evaluation_manifest.json"
    protected.write_text('{"tiers":{"E3":{"sample_ids":["protected-only"]}}}', "utf-8")
    config = {
        "common_image_root": str(common_root),
        "seed": 42,
        "training_selection_mode": "cyclic_full_coverage",
        "cycle": {"protected_evaluation_manifest": str(protected)},
        "sources": [
            {
                "name": "VRSBench",
                "image_root": str(vrs_root),
                "train_file": str(vrs_train),
                "validation_file": str(vrs_val),
                "training_task_quotas": {"detection": 2, "vqa": 1, "counting": 1},
            },
            {
                "name": "LEVIR-CC",
                "image_root": str(levir_root),
                "train_file": str(levir_train),
                "validation_file": str(levir_val),
                "training_samples_per_image_group": 2,
            },
        ],
    }

    first = module.prepare_training_cycle(config, cycle_output_dir=tmp_path / "cycle-a")
    second = module.prepare_training_cycle(config, cycle_output_dir=tmp_path / "cycle-b")

    assert first["num_rounds"] == 3
    assert first["global"]["valid"] is True
    assert first["global"]["population_samples"] == 10
    assert first["protected_evaluation"]["overlap_count"] == 0
    assert [item["sha256"] for item in first["rounds"]] == [
        item["sha256"] for item in second["rounds"]
    ]


def test_full_cycle_spreads_and_replays_short_source_to_match_pattern(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    common_root = tmp_path / "datasets"
    vrs_root = common_root / "VRSBench"
    levir_root = common_root / "LEVIR-CC"
    vrs_image = vrs_root / "images/vrs.png"
    vrs_image.parent.mkdir(parents=True)
    vrs_image.write_bytes(b"png")
    pair_images: list[str] = []
    val_pair_images: list[str] = []
    for split, destination in (("train", pair_images), ("val", val_pair_images)):
        for period in ("A", "B"):
            image = levir_root / "images" / split / period / "pair.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"png")
            destination.append(f"images/{split}/{period}/pair.png")

    vrs_train = vrs_root / "train.jsonl"
    vrs_val = vrs_root / "val.jsonl"
    write_jsonl(
        vrs_train,
        [
            internal_row(f"vrs-{index}", ["images/vrs.png"], task_type="detection")
            for index in range(8)
        ],
    )
    write_jsonl(vrs_val, [internal_row("vrs-val", ["images/vrs.png"], task_type="vqa")])
    levir_train = levir_root / "train.jsonl"
    levir_val = levir_root / "val.jsonl"
    write_jsonl(
        levir_train,
        [internal_row(f"levir-{index}", pair_images) for index in range(4)],
    )
    write_jsonl(levir_val, [internal_row("levir-val", val_pair_images)])
    protected = tmp_path / "evaluation_manifest.json"
    protected.write_text('{"tiers":{"E3":{"sample_ids":["protected-only"]}}}', "utf-8")
    config = {
        "common_image_root": str(common_root),
        "seed": 42,
        "training_selection_mode": "cyclic_full_coverage",
        "cycle": {
            "protected_evaluation_manifest": str(protected),
            "source_batch_pattern": ["VRSBench", "VRSBench", "VRSBench", "LEVIR-CC"],
            "replay_short_source": {
                "enabled": True,
                "source": "LEVIR-CC",
                "reference_source": "VRSBench",
            },
        },
        "sources": [
            {
                "name": "VRSBench",
                "image_root": str(vrs_root),
                "train_file": str(vrs_train),
                "validation_file": str(vrs_val),
                "training_task_quotas": {"detection": 2},
            },
            {
                "name": "LEVIR-CC",
                "image_root": str(levir_root),
                "train_file": str(levir_train),
                "validation_file": str(levir_val),
                "training_samples_per_image_group": 2,
            },
        ],
    }

    report = module.prepare_training_cycle(config, cycle_output_dir=tmp_path / "cycle")

    assert report["num_rounds"] == 4
    assert report["global"]["valid"] is True
    assert report["global"]["population_samples"] == 12
    assert report["global"]["duplicate_count"] == 0
    assert report["source_scheduling"]["replay"]["replay_exposures_added"] == 2
    assert report["source_scheduling"]["expected_exposure_counts"] == {
        "LEVIR-CC": 6,
        "VRSBench": 8,
    }
    for round_report in report["rounds"]:
        assert round_report["source_distribution"]["VRSBench"] == 2
        assert round_report["source_distribution"]["LEVIR-CC"] >= 1
    replay_rows = [
        row
        for round_report in report["rounds"]
        for row in read_jsonl(round_report["train_file"])
        if row["metadata"].get("cycle_replay")
    ]
    assert len(replay_rows) == 2
    assert all("replay_original_id" in row["metadata"] for row in replay_rows)
    assert len({row["id"] for row in replay_rows}) == 2


def test_stage_a_prompt_profile_strengthens_existing_messages() -> None:
    module = load_script_module()
    sample = {
        "id": "count-1",
        "task_type": "counting",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "How many cars?"}]},
            {"role": "assistant", "content": '{"count":3}'},
        ],
    }
    converted = module._to_qwen_row(
        sample,
        source_name="VRSBench",
        instruction_override=None,
        prompt_profile="qwen3vl_4b_stage_a",
        strengthen_existing_messages=True,
    )
    assert "ONLY the integer" in converted["messages"][0]["content"][0]["text"]
    assert converted["messages"][1]["content"] == "3"
