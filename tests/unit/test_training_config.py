from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sat_rs_vlm.training.config import load_training_config


def test_training_config_loads() -> None:
    config = load_training_config("configs/train/qwen3vl_lora.yaml")
    assert config.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.training.method == "qlora"
    assert config.training.freeze_vision_encoder is True
    assert config.training.dataloader_num_workers == 0
    assert config.training.dataloader_persistent_workers is False
    assert config.lora.r == 16
    assert config.qlora.load_in_4bit is True
    assert config.loss.mode == "task_weighted"


def test_h1_training_sections_are_parsed() -> None:
    config = load_training_config(
        "configs/train/qwen3vl_hard_visual_adaptation.yaml",
        allow_unresolved_env=True,
    )

    assert config.loss.mode == "task_weighted"
    assert config.statistics.enabled is True
    assert config.hard_adaptation.enabled is True
    assert config.vision_tuning.enabled is True
    assert config.training.target_effective_epochs == pytest.approx(1.5)


def test_unknown_training_field_is_not_silently_ignored(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/train/qwen3vl_local_smoke.yaml").read_text("utf-8"))
    payload["training"]["misspelled_loss_mode"] = "token_mean"
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="misspelled_loss_mode"):
        load_training_config(config_path, allow_unresolved_env=True)


def test_unknown_top_level_section_is_not_silently_ignored(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/train/qwen3vl_local_smoke.yaml").read_text("utf-8"))
    payload["losss"] = {"mode": "token_mean"}
    config_path = tmp_path / "invalid-top-level.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="losss"):
        load_training_config(config_path, allow_unresolved_env=True)


def test_statistics_and_hard_mining_share_bbox_thresholds(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/train/qwen3vl_hard_visual_adaptation.yaml").read_text("utf-8")
    )
    payload["hard_adaptation"]["bbox_area_thresholds"]["small_max"] = 0.02
    config_path = tmp_path / "mismatched-thresholds.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="identical bbox_area_thresholds"):
        load_training_config(config_path, allow_unresolved_env=True)


def test_h2_a_config_strictly_preserves_single_variable_protocol() -> None:
    config = load_training_config(
        "configs/train/qwen3vl_h2_global_refinement.yaml",
        allow_unresolved_env=True,
    )

    assert config.h2_refinement.enabled is True
    assert config.h2_refinement.difficulty_mode == "cell_rank"
    assert config.h2_refinement.source_weights == {"VRSBench": 0.75, "LEVIR-CC": 0.25}
    assert config.loss.mode == "task_weighted"
    assert set(config.loss.task_weights.values()) == {1.0}
    assert config.vision_tuning.enabled is False
    assert config.training.freeze_vision_encoder is True
    assert config.data.sampling_mode == "uniform"
    assert config.training.num_train_epochs is None
    assert config.training.max_steps is None
    assert config.training.target_effective_epochs == pytest.approx(1.5)
    assert config.training.max_effective_epochs == pytest.approx(2.0)
    assert config.training.allow_overtrain is False
    assert config.lora.initial_adapter_dir == config.h2_refinement.source_checkpoint


def test_h2_a_requires_replay_adapter_and_rejects_vision_tuning(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        Path("configs/train/qwen3vl_h2_global_refinement.yaml").read_text("utf-8")
    )
    payload["lora"]["initial_adapter_dir"] = None
    payload["vision_tuning"]["enabled"] = True
    config_path = tmp_path / "invalid-h2.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="Invalid H2-A configuration"):
        load_training_config(config_path, allow_unresolved_env=True)


def test_qwen3vl_4b_future_training_configs_are_isolated() -> None:
    smoke = load_training_config(
        "configs/train/qwen3vl_4b_lora_smoke.yaml",
        allow_unresolved_env=True,
    )
    full = load_training_config(
        "configs/train/qwen3vl_4b_lora_4090.yaml",
        allow_unresolved_env=True,
    )
    h2 = load_training_config(
        "configs/train/qwen3vl_4b_h2_global_refinement_4090.yaml",
        allow_unresolved_env=True,
    )

    for config in (smoke, full, h2):
        assert config.model.model_dir == "${QWEN3VL_4B_MODEL_DIR}"
        assert "qwen3vl_4b" in config.training.output_dir
        assert config.training.method == "lora"
        assert config.vision_tuning.enabled is False
    assert smoke.training.max_steps == 2
    assert full.training.per_device_train_batch_size == 4
    assert h2.lora.initial_adapter_dir == "${QWEN3VL_4B_REPLAY_ADAPTER_DIR}"
    assert h2.h2_refinement.source_checkpoint == h2.lora.initial_adapter_dir
    assert h2.h2_refinement.output_dir == "data/processed/h2/qwen3vl_4b"


def test_qwen3vl_4b_stage_a_uses_strict_full_coverage_contract() -> None:
    config = load_training_config(
        "configs/train/qwen3vl_4b_stage_a_multisource_4090.yaml",
        allow_unresolved_env=True,
    )

    assert config.model.model_dir == config.model.processor_dir
    assert config.cycle_training.enabled is True
    assert config.cycle_training.selection_mode == "cyclic_full_coverage"
    assert config.cycle_training.learning_rates == [2.0e-5, 1.0e-5]
    assert config.data.sampling_mode == "alternating_source"
    assert config.data.source_exhaustion_policy == "coverage_first"
    assert config.training.per_device_train_batch_size == 4
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.num_train_epochs == 1
    assert config.training.max_steps is None
    assert config.training.freeze_vision_encoder is True
    assert config.vision_tuning.enabled is False
    assert config.loss.mode == "task_weighted"
    assert set(config.loss.task_weights.values()) == {1.0}
