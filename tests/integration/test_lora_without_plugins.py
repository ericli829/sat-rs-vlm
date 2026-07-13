from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_lora_help_works_with_missing_plugin_root(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["SAT_RS_VLM_PLUGIN_ROOT"] = str(tmp_path / "deleted-plugin-root")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_qwen3vl_lora.py"), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
