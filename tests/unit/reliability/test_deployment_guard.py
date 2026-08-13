from pathlib import Path

import pytest

from sat_rs_vlm.models.reliability.adapter_redundancy import initialize_adapter_replicas
from sat_rs_vlm.models.reliability.deployment_guard import (
    PeriodicAdapterGuard,
    guard_adapter_before_inference,
)


def _adapter(root: Path, content: bytes = b"weights") -> None:
    root.mkdir()
    (root / "adapter_model.safetensors").write_bytes(content)
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_guard_restores_working_adapter_before_inference(tmp_path: Path) -> None:
    working, warm, golden = tmp_path / "working", tmp_path / "warm", tmp_path / "golden"
    _adapter(working)
    _adapter(warm)
    _adapter(golden)
    manifest = tmp_path / "manifest.json"
    initialize_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest_path=manifest)
    (working / "adapter_model.safetensors").write_bytes(b"bad")

    result = guard_adapter_before_inference(working, warm_adapter=warm, golden_adapter=golden, manifest=manifest)

    assert result.success
    assert (working / "adapter_model.safetensors").read_bytes() == b"weights"


def test_guard_fails_closed_without_trusted_recovery_source(tmp_path: Path) -> None:
    working, warm, golden = tmp_path / "working", tmp_path / "warm", tmp_path / "golden"
    _adapter(working)
    _adapter(warm)
    _adapter(golden)
    manifest = tmp_path / "manifest.json"
    initialize_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest_path=manifest)
    for root in (working, warm, golden):
        (root / "adapter_model.safetensors").write_bytes(b"bad")

    with pytest.raises(RuntimeError, match="blocked inference"):
        guard_adapter_before_inference(working, warm_adapter=warm, golden_adapter=golden, manifest=manifest)



def test_periodic_guard_scrubs_on_configured_interval(tmp_path: Path) -> None:
    working, warm, golden = tmp_path / "working", tmp_path / "warm", tmp_path / "golden"
    _adapter(working)
    _adapter(warm)
    _adapter(golden)
    manifest = tmp_path / "manifest.json"
    initialize_adapter_replicas(working, warm_root=warm, golden_root=golden, manifest_path=manifest)
    guard = PeriodicAdapterGuard(
        working, warm_adapter=warm, golden_adapter=golden, manifest=manifest, interval_batches=2
    )
    assert guard.after_batch() is None
    (working / "adapter_model.safetensors").write_bytes(b"bad")
    result = guard.after_batch()

    assert result is not None and result.success
    assert guard.report()["events"][0]["action"] == "restored"
