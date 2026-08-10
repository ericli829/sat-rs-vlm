from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "evaluation" / "plot_evaluation_results.py"


def test_plot_evaluation_cli(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    output = tmp_path / "plots"
    evaluation.mkdir()
    summary = {
        "contract_version": "1.5",
        "overall": {
            "metrics": {},
            "latency_context": {"status": "unresolved"},
            "task_distribution": {"captioning": 2},
        },
        "by_task": {
            "captioning": {
                "metrics": {
                    "bleu_1_approx": {
                        "value": 0.5,
                        "label": "internal",
                        "status": "ok",
                        "num_samples": 2,
                        "note": None,
                    },
                    "length_ratio": {
                        "value": 1.0,
                        "label": "internal",
                        "status": "ok",
                        "num_samples": 2,
                        "note": None,
                    },
                }
            }
        },
        "by_qa_type": {},
        "semantic": {},
    }
    (evaluation / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "plotting": {
                    "evaluations": [f"model={evaluation}"],
                    "comparisons": [],
                    "formats": ["png"],
                },
                "output": {"figures_dir": str(output)},
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "task_sample_distribution" in payload["generated"]
    assert (output / "plot_manifest.json").is_file()
