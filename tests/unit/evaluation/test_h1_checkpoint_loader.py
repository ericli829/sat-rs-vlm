from __future__ import annotations

import json
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


def test_generic_visual_sidecar_manifest_checksum_is_enforced(tmp_path: Path) -> None:
    from sat_rs_vlm.models.reliability.checksum import file_sha256

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "processor").mkdir()
    sidecar = checkpoint / "visual_trainable_weights.safetensors"
    sidecar.write_bytes(b"visual")
    (checkpoint / "visual_trainable_manifest.json").write_text(
        json.dumps({"weights": sidecar.name, "sha256": file_sha256(sidecar)}),
        encoding="utf-8",
    )
    manifest = {
        "strategy": "lora",
        "adapter_based": True,
        "checkpoint_type": "adapter_with_visual_sidecar",
        "visual_sidecar": sidecar.name,
    }
    validate_checkpoint_files(checkpoint, manifest)
    sidecar.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_checkpoint_files(checkpoint, manifest)
