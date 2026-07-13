from pathlib import Path

import pytest

from sat_rs_vlm.plugins import PluginCompatibilityError, capability
from sat_rs_vlm.plugins.manifest import load_plugin_manifest


def test_cuda_requirement_fails_without_loading_model(
    fake_plugin_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_plugin_manifest(fake_plugin_root / "plugins" / "fake_strategy")
    manifest.compatibility.requires_cuda = True
    monkeypatch.setattr(
        capability,
        "probe_cuda",
        lambda timeout_seconds=10: {
            "status": "probe_failed",
            "cuda_available": False,
        },
    )

    with pytest.raises(PluginCompatibilityError, match="CUDA is required"):
        capability.validate_platform_capability(manifest)
