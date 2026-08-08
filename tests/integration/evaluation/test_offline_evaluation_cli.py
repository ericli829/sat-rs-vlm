from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_offline_evaluation_cli(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / "evaluation"
    command = [
        sys.executable,
        str(project_root / "scripts" / "evaluation" / "evaluate_predictions.py"),
        "--predictions",
        str(project_root / "tests" / "fixtures" / "evaluation_v1_5" / "predictions.jsonl"),
        "--output-dir",
        str(output_dir),
        "--no-semantic",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "evaluated_predictions.jsonl").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["contract_version"] == "1.5"
    assert summary["overall"]["metrics"]["num_samples"]["value"] == 10
