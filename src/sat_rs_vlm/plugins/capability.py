"""Lightweight platform probes used before an external plugin loads model code."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import Any

from sat_rs_vlm.plugins.errors import PluginCompatibilityError
from sat_rs_vlm.plugins.manifest import ExternalPluginManifest


def probe_cuda(timeout_seconds: int = 10) -> dict[str, Any]:
    """Probe Torch/CUDA in a subprocess so a broken runtime cannot hang the caller."""

    command = [
        sys.executable,
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({'torch': torch.__version__, "
            "'cuda_available': torch.cuda.is_available(), "
            "'cuda_version': torch.version.cuda, "
            "'device_count': torch.cuda.device_count()}))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout", "cuda_available": False}
    if completed.returncode != 0:
        return {
            "status": "probe_failed",
            "cuda_available": False,
            "error": (completed.stderr or completed.stdout).strip()[-1000:],
        }
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {
            "status": "invalid_probe_output",
            "cuda_available": False,
            "output": completed.stdout.strip()[-1000:],
        }
    if not isinstance(payload, dict):
        return {"status": "invalid_probe_output", "cuda_available": False}
    return {"status": "ok", **payload}


def validate_platform_capability(manifest: ExternalPluginManifest) -> dict[str, Any]:
    """Validate operating-system and optional CUDA requirements from a manifest."""

    current = platform.system().lower()
    allowed = {item.lower() for item in manifest.compatibility.platforms}
    if current not in allowed:
        raise PluginCompatibilityError(
            plugin_name=manifest.plugin.name,
            stage="platform",
            reason=f"platform {current!r} is not in {sorted(allowed)}",
            suggested_action="Use a supported platform or update the verified manifest.",
        )
    result: dict[str, Any] = {"platform": current, "cuda": None}
    if manifest.compatibility.requires_cuda:
        cuda = probe_cuda()
        result["cuda"] = cuda
        if not bool(cuda.get("cuda_available")):
            raise PluginCompatibilityError(
                plugin_name=manifest.plugin.name,
                stage="device",
                reason=f"CUDA is required but unavailable ({cuda.get('status')})",
                suggested_action="Use a CUDA-enabled project or plugin-specific environment.",
            )
    return result
