from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.training.config import HardAdaptationConfig
from sat_rs_vlm.training.hard_example_mining import (
    build_hard_example_dataset,
    categorize_difficulty,
    load_evaluation_ids,
    load_rows,
    score_hard_example,
    select_stratified_samples,
)


def _training_row(sample_id: str, task: str, source: str = "VRSBench") -> dict[str, object]:
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "task_type": task,
        "metadata": {"dataset": source, "training_source": source},
    }


def _evaluated(sample_id: str, task: str, metrics: dict[str, object]) -> dict[str, object]:
    reference = '{"label":"car","bbox":[0.1,0.1,0.15,0.15]}' if task == "detection" else "2"
    return {
        "id": sample_id,
        "task_type": task,
        "prediction": "wrong",
        "reference": reference,
        "parse_ok": True,
        "sample_metrics": metrics,
        "metadata": {},
    }


def test_hard_score_is_reproducible_and_preserves_reasons() -> None:
    config = HardAdaptationConfig(fixed_evaluation_sample_count=1)
    row = _evaluated(
        "det",
        "detection",
        {
            "parse_success": True,
            "valid_coordinate": True,
            "label_match": False,
            "iou": 0.1,
            "generalized_iou": -0.2,
            "normalized_center_distance": 0.5,
        },
    )

    first = score_hard_example(row, config)
    second = score_hard_example(row, config)

    assert first == second
    assert "low_iou" in first["hard_reason"]
    assert "label_error" in first["hard_reason"]
    assert "small_object" in first["hard_reason"]
    assert first["hard_diagnostics"]["bbox_area_bucket"] == "small"


def test_build_excludes_eval_ids_and_replay_is_seed_reproducible(tmp_path: Path) -> None:
    rows = [
        _training_row("eval-only", "detection"),
        _training_row("hard-det", "detection"),
        _training_row("hard-count", "counting"),
        _training_row("replay-det", "detection"),
        _training_row("replay-count", "counting"),
        _training_row("replay-vqa", "vqa"),
        _training_row("replay-caption", "captioning"),
        _training_row("replay-scene", "scene_classification"),
        _training_row("replay-change", "change_detection", "LEVIR-CC"),
    ]
    evaluated = [
        _evaluated(
            "eval-only",
            "detection",
            {
                "parse_success": False,
                "valid_coordinate": False,
                "label_match": False,
                "iou": 0.0,
                "normalized_center_distance": 1.0,
            },
        ),
        _evaluated(
            "hard-det",
            "detection",
            {
                "parse_success": True,
                "valid_coordinate": True,
                "label_match": False,
                "iou": 0.0,
                "normalized_center_distance": 0.8,
            },
        ),
        _evaluated(
            "hard-count",
            "counting",
            {
                "number_parse_success": True,
                "exact_count_correct": False,
                "within_1_correct": False,
                "absolute_error": 3,
            },
        ),
    ]
    config = HardAdaptationConfig(
        fixed_evaluation_sample_count=1,
        hard_score_threshold=0.2,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build_hard_example_dataset(
        rows,
        evaluated,
        {"eval-only"},
        config,
        seed=42,
        output_dir=first_dir,
        prediction_source="train-mining-eval",
        source_checkpoint="stage-b",
    )
    second = build_hard_example_dataset(
        rows,
        evaluated,
        {"eval-only"},
        config,
        seed=42,
        output_dir=second_dir,
        prediction_source="train-mining-eval",
        source_checkpoint="stage-b",
    )

    assert first["regular_replay_ids"] == second["regular_replay_ids"]
    all_ids = {str(row["id"]) for row in load_rows(first_dir / "h1_train.jsonl")}
    assert "eval-only" not in all_ids
    assert first["excluded_evaluation_id_count"] == 1
    assert "fixed 1-sample evaluation set" in first["evaluation_exclusion_statement"]
    assert first["checksums"]["h1_train.jsonl"] == second["checksums"]["h1_train.jsonl"]


def test_tier_manifest_uses_e3_as_fail_closed_exclusion_set(tmp_path: Path) -> None:
    manifest = {"tiers": {"E1": {"sample_ids": ["e1"]}, "E3": {"sample_ids": ["e1", "e3"]}}}
    path = tmp_path / "evaluation_tiers_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_evaluation_ids(path) == {"e1", "e3"}


def test_h2_reusable_interfaces_require_explicit_thresholds_and_are_deterministic() -> None:
    scored = [
        {"id": "regular", "hard_score": 0.1},
        {"id": "medium", "hard_score": 0.4},
        {"id": "core", "hard_score": 0.8},
    ]
    categories = categorize_difficulty(
        scored,
        medium_hard_threshold=0.3,
        core_hard_threshold=0.7,
    )

    assert [row["id"] for row in categories["regular_representative"]] == ["regular"]
    assert [row["id"] for row in categories["medium_hard"]] == ["medium"]
    assert [row["id"] for row in categories["core_hard"]] == ["core"]
    candidates = [
        _training_row("d1", "detection"),
        _training_row("d2", "detection"),
        _training_row("c1", "counting"),
        _training_row("c2", "counting"),
    ]
    assert select_stratified_samples(candidates, 2, seed=42) == select_stratified_samples(
        candidates, 2, seed=42
    )
    with pytest.raises(ValueError, match="0 <= medium < core"):
        categorize_difficulty(scored, medium_hard_threshold=0.8, core_hard_threshold=0.7)
