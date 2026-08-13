from pathlib import Path

import yaml

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
    assert set(config.loss.task_weights.values()) == {1.0}
    assert config.data.task_sampling_weights is not config.loss.task_weights


def test_legacy_config_defaults_to_task_weighted_and_can_select_token_mean(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(Path("configs/train/qwen3vl_local_smoke.yaml").read_text("utf-8"))
    source.pop("loss", None)
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    assert load_training_config(legacy, allow_unresolved_env=True).loss.mode == "task_weighted"

    source["loss"] = {"mode": "token_mean"}
    historical = tmp_path / "historical.yaml"
    historical.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    assert load_training_config(historical, allow_unresolved_env=True).loss.mode == "token_mean"
