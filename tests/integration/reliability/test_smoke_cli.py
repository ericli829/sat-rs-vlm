import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_smoke_cli_all_generates_required_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    environment = dict(os.environ)
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/reliability/run_smoke.py"),
            "--case",
            "all",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "cli",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    run_dir = Path(payload["run_dir"])
    assert payload["execution_mode"] == "smoke_mock"
    assert (run_dir / "config_resolved.yaml").is_file()
    assert (run_dir / "faults/injection_records.jsonl").is_file()
    assert (run_dir / "metrics/summary.json").is_file()
    assert (run_dir / "protection/strategy_results.json").is_file()
