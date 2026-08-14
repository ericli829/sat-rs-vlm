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
