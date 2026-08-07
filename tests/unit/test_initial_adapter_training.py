from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.train_qwen3vl_lora import apply_lora, prune_training_checkpoints

from sat_rs_vlm.training.config import ResolvedTrainingPaths


def test_apply_lora_loads_existing_adapter_as_trainable(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    paths = ResolvedTrainingPaths(
        model_source="base",
        processor_source="base",
        model_dir=None,
        processor_dir=None,
        train_file=tmp_path / "train.jsonl",
        val_file=tmp_path / "val.jsonl",
        image_root=tmp_path,
        output_dir=tmp_path / "output",
        initial_adapter_dir=adapter,
    )
    calls: dict[str, Any] = {}

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: Any, path: str, *, is_trainable: bool) -> Any:
            calls.update(model=model, path=path, is_trainable=is_trainable)
            return "trainable-adapter"

    config = SimpleNamespace(training=SimpleNamespace(method="lora"))
    result = apply_lora(
        "base-model",
        config,
        paths,
        {"peft": SimpleNamespace(PeftModel=FakePeftModel)},
    )

    assert result == "trainable-adapter"
    assert calls == {
        "model": "base-model",
        "path": str(adapter),
        "is_trainable": True,
    }


def test_prune_training_checkpoints_keeps_newest_resume_points(tmp_path: Path) -> None:
    output = tmp_path / "output"
    for step in (500, 1000, 1500, 2000):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    unrelated = output / "processor"
    unrelated.mkdir()

    removed = prune_training_checkpoints(output, keep=2)

    assert {path.name for path in removed} == {"checkpoint-500", "checkpoint-1000"}
    assert (output / "checkpoint-1500").is_dir()
    assert (output / "checkpoint-2000").is_dir()
    assert unrelated.is_dir()
