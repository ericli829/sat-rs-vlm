from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_plugin_listing_does_not_modify_lora_script(fake_plugin_root: Path) -> None:
    lora_script = PROJECT_ROOT / "scripts" / "train_qwen3vl_lora.py"
    before = hashlib.sha256(lora_script.read_bytes()).hexdigest()
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "list_external_plugins.py"),
            "--plugin-root",
            str(fake_plugin_root),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fake_strategy" in result.stdout
    assert hashlib.sha256(lora_script.read_bytes()).hexdigest() == before
