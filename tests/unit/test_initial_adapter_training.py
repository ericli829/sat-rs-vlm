from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.train_qwen3vl_lora import apply_lora

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
