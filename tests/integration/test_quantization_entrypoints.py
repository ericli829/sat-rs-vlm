from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return dict(json.loads(completed.stdout))


def test_quantization_benchmark_config_dry_run(tmp_path: Path) -> None:
    report = _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "quantize_rs_vlm.py"),
            "--config",
            str(
                PROJECT_ROOT / "configs" / "quantization" / "qwen3vl_torch_dynamic_int8_smoke.yaml"
            ),
            "--output-dir",
            str(tmp_path / "quantization"),
            "--dry-run",
        ]
    )
    assert report["success"] is True
    assert report["backend"] == "torch_dynamic_int8"


def test_quantization_sensitivity_config_dry_run(tmp_path: Path) -> None:
    report = _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "quantization_sensitivity_test.py"),
            "--config",
            str(PROJECT_ROOT / "configs" / "quantization" / "quantization_sensitivity_smoke.yaml"),
            "--output-dir",
            str(tmp_path / "sensitivity"),
            "--dry-run",
        ]
    )
    assert report["success"] is True
    assert report["requested_samples"] == 2
