from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_base_multidataset import _metric_values, _write_summary


def test_multidataset_summary_extracts_vrsbench_and_levir_metrics(tmp_path: Path) -> None:
    payload = {
        "by_task": {
            "vqa": {
                "metrics": {
                    "normalized_accuracy": {"value": 0.75},
                    "keyword_hit": {"value": 1.0},
                }
            }
        },
        "by_protocol": {
            "levir_cc_change_caption": {
                "metrics": {
                    "balanced_accuracy": {"value": 0.8},
                    "change_f1": {"value": 0.7},
                }
            }
        },
    }

    metrics = _metric_values(payload)

    assert metrics["by_task.vqa.metrics.normalized_accuracy"] == 0.75
    assert metrics["by_protocol.levir_cc_change_caption.metrics.change_f1"] == 0.7
    assert all("keyword_hit" not in name for name in metrics)

    report = {
        "schema_version": "1.0",
        "model_source": "base-model",
        "runs": {
            "vrsbench": {
                "sample_count": 1,
                "output_dir": "vrsbench",
                "metrics": metrics,
            }
        },
    }
    _write_summary(report, tmp_path)

    written = json.loads((tmp_path / "base_multidataset_summary.json").read_text(encoding="utf-8"))
    assert written["model_source"] == "base-model"
    assert (tmp_path / "base_multidataset_summary.md").is_file()
