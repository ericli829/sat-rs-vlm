from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.evaluation.checkpoint_loader import validate_checkpoint_files


def test_h1_manifest_requires_visual_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "processor").mkdir()
    manifest = {
        "strategy": "lora",
        "adapter_based": True,
        "checkpoint_type": "adapter_with_visual_sidecar",
        "visual_sidecar": "h1_visual_weights.safetensors",
    }

    with pytest.raises(FileNotFoundError, match="visual sidecar"):
        validate_checkpoint_files(checkpoint, manifest)

    (checkpoint / "h1_visual_weights.safetensors").write_bytes(b"visual")
    validate_checkpoint_files(checkpoint, manifest)
