from pathlib import Path

from sat_rs_vlm.models.reliability.adapter_redundancy import (
    initialize_adapter_replicas,
    scrub_adapter_replicas,
)


def _adapter(root: Path) -> None:
    root.mkdir()
    (root / "adapter_model.safetensors").write_bytes(b"weights")
    (root / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    processor = root / "processor"
    processor.mkdir()
    (processor / "config.json").write_text("processor", encoding="utf-8")


def test_adapter_scrub_recovers_weight_and_config_from_warm(tmp_path: Path) -> None:
    working, warm, golden = tmp_path / "working", tmp_path / "warm", tmp_path / "golden"
    _adapter(working)
    warm.mkdir()
    golden.mkdir()
    for destination in (warm, golden):
        (destination / "adapter_model.safetensors").write_bytes(b"weights")
        (destination / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        (destination / "processor").mkdir()
        (destination / "processor/config.json").write_text("processor", encoding="utf-8")
    manifest_path = tmp_path / "adapter_manifest.json"
    initialize_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest_path=manifest_path)
    (working / "adapter_model.safetensors").write_bytes(b"bad")
    (working / "adapter_config.json").write_text("bad", encoding="utf-8")

    result = scrub_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest=manifest_path)

    assert result.success
    assert set(result.restored_from_warm) == {"adapter_model.safetensors", "adapter_config.json"}
    assert (working / "adapter_model.safetensors").read_bytes() == b"weights"


def test_adapter_scrub_refuses_untrusted_warm_and_uses_golden(tmp_path: Path) -> None:
    working, warm, golden = tmp_path / "working", tmp_path / "warm", tmp_path / "golden"
    _adapter(working)
    for destination in (warm, golden):
        _adapter(destination)
    manifest_path = tmp_path / "adapter_manifest.json"
    initialize_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest_path=manifest_path)
    (working / "adapter_model.safetensors").write_bytes(b"bad")
    (warm / "adapter_model.safetensors").write_bytes(b"also-bad")

    result = scrub_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest=manifest_path)

    assert result.success
    assert result.restored_from_golden == ["adapter_model.safetensors"]
