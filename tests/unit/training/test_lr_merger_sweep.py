from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.training.lr_merger_sweep import (
    ExperimentSpec,
    build_evaluation_command,
    materialize_training_config,
    select_phase_a_lr,
    validate_probe_contract,
)


def test_evaluation_command_forwards_batch_override() -> None:
    command = build_evaluation_command(
        evaluation_config="configs/eval/qwen3vl_4b_e1_sweep_4090.yaml",
        checkpoint="checkpoints/round1",
        output_dir="outputs/evaluation_e1",
        batch_size=4,
    )
    assert command[-2:] == ["--batch-size", "4"]


def test_phase_a_selector_uses_weak_score_and_guard() -> None:
    baseline = {
        "metrics": {
            "parse_success": 1.0,
            "detection_miou": 0.60,
            "levir_f1": 0.80,
            "caption_rouge_l": 0.40,
        }
    }
    candidates = [
        {
            "experiment_id": "A1",
            "status": "ANALYZED",
            "lora_lr": 2e-5,
            "metrics": {
                "parse_success": 1.0,
                "detection_miou": 0.60,
                "levir_f1": 0.80,
                "caption_rouge_l": 0.40,
                "count_exact": 0.50,
                "count_pm1": 0.70,
                "scene_normalized": 0.50,
                "vqa_normalized": 0.50,
            },
        },
        {
            "experiment_id": "A2",
            "status": "ANALYZED",
            "lora_lr": 5e-5,
            "metrics": {
                "parse_success": 1.0,
                "detection_miou": 0.61,
                "levir_f1": 0.80,
                "caption_rouge_l": 0.40,
                "count_exact": 0.60,
                "count_pm1": 0.80,
                "scene_normalized": 0.70,
                "vqa_normalized": 0.70,
            },
        },
    ]
    selected = select_phase_a_lr(candidates, baseline)
    assert selected["selected_lr"] == pytest.approx(5e-5)
    assert selected["selection_mode"] == "weak_score_with_guards"


def test_phase_a_selector_falls_back_when_all_candidates_break_guard() -> None:
    baseline = {
        "metrics": {
            "parse_success": 1.0,
            "detection_miou": 0.70,
            "levir_f1": 0.80,
            "caption_rouge_l": 0.50,
        }
    }
    candidate = {
        "experiment_id": "A3",
        "status": "ANALYZED",
        "lora_lr": 1e-4,
        "metrics": {
            "parse_success": 0.98,
            "detection_miou": 0.50,
            "levir_f1": 0.70,
            "caption_rouge_l": 0.40,
            "count_exact": 0.9,
            "count_pm1": 0.9,
            "scene_normalized": 0.9,
            "vqa_normalized": 0.9,
        },
    }
    selected = select_phase_a_lr([candidate], baseline)
    assert selected["selected_lr"] == pytest.approx(5e-5)
    assert selected["selection_mode"] == "fallback_due_to_guard_failure"


def test_probe_contract_checks_sha_and_protected_ids(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps({"id": "train-1", "task_type": "captioning"}) + "\n",
        encoding="utf-8",
    )
    import hashlib

    digest = hashlib.sha256(train.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"output_sha256": digest, "total_samples": 1}), encoding="utf-8"
    )
    protected = tmp_path / "tiers.json"
    protected.write_text(
        json.dumps({"tiers": {"E1": {"sample_ids": ["eval-1"]}}}), encoding="utf-8"
    )
    result = validate_probe_contract(train, manifest, protected)
    assert result["sha256"] == digest
    assert result["protected_overlap_count"] == 0


def test_materialized_merger_config_uses_existing_training_schema(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    base = root / "configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml"
    destination = tmp_path / "resolved.yaml"
    payload = materialize_training_config(
        base,
        output_path=destination,
        model_dir=tmp_path / "model",
        processor_dir=tmp_path / "model",
        train_file=tmp_path / "train.jsonl",
        val_file=tmp_path / "val.jsonl",
        image_root=tmp_path,
        initial_adapter=tmp_path / "round1",
        output_dir=tmp_path / "checkpoint",
        spec=ExperimentSpec("B1", "merger", "B", 2e-5, 1e-5, 0, True),
        common={
            "max_seq_length": 1024,
            "seed": 42,
            "bf16": True,
            "gradient_checkpointing": True,
        },
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=8,
    )
    assert payload["vision_tuning"]["unfreeze_last_n_blocks"] == 0
    assert payload["vision_tuning"]["train_main_merger"] is True
    assert destination.is_file()


def test_lora_only_materialization_uses_unused_positive_merger_placeholder(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    base = root / "configs/train/qwen3vl_4b_vit_probe_last2_4090.yaml"
    destination = tmp_path / "lora_only.yaml"
    payload = materialize_training_config(
        base,
        output_path=destination,
        model_dir=tmp_path / "model",
        processor_dir=tmp_path / "model",
        train_file=tmp_path / "train.jsonl",
        val_file=tmp_path / "val.jsonl",
        image_root=tmp_path,
        initial_adapter=tmp_path / "round1",
        output_dir=tmp_path / "checkpoint",
        spec=ExperimentSpec("A1", "lora-only", "A", 2e-5, 0.0, 0, False),
        common={"max_seq_length": 1024, "seed": 42},
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
    )
    assert payload["optimization"]["visual_merger_lr"] == pytest.approx(1.0e-6)
    assert payload["vision_tuning"]["enabled"] is False
    assert payload["vision_tuning"]["train_main_merger"] is False
