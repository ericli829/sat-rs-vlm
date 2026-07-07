from pathlib import Path

import pytest

from sat_rs_vlm.training.config import (
    TrainingPathOverrides,
    apply_training_overrides,
    load_training_config,
    resolve_path,
    resolve_training_paths,
)


def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "train.yaml"
    monkeypatch.setenv("LOCAL_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("TRAIN_JSONL", str(tmp_path / "train.jsonl"))
    monkeypatch.setenv("VAL_JSONL", str(tmp_path / "val.jsonl"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    config_path.write_text(
        """
model:
  model_dir: "${LOCAL_MODEL_DIR}"
  processor_dir: "${LOCAL_MODEL_DIR}"
data:
  train_file: "${TRAIN_JSONL}"
  val_file: "${VAL_JSONL}"
  image_root: "${DATA_ROOT}"
training:
  output_dir: "out"
lora: {}
""",
        encoding="utf-8",
    )
    config = load_training_config(config_path)
    assert config.model.model_dir == str(tmp_path / "model")
    assert config.data.train_file == str(tmp_path / "train.jsonl")


def test_cli_override_replaces_unresolved_env(tmp_path: Path) -> None:
    config = load_training_config(
        "configs/train/qwen3vl_local_smoke.yaml",
        allow_unresolved_env=True,
    )
    config = apply_training_overrides(
        config,
        TrainingPathOverrides(
            model_dir=str(tmp_path / "model"),
            train_file=str(tmp_path / "train.jsonl"),
            val_file=str(tmp_path / "val.jsonl"),
            image_root=str(tmp_path),
        ),
    )
    paths = resolve_training_paths(config)
    assert paths.model_dir == tmp_path / "model"
    assert paths.train_file == tmp_path / "train.jsonl"


def test_relative_path_resolution(tmp_path: Path) -> None:
    assert resolve_path("data/train.jsonl", tmp_path) == tmp_path / "data" / "train.jsonl"
