from __future__ import annotations

import json
from pathlib import Path

import scripts.evaluate_taskgraph as evaluate_taskgraph

from sat_rs_vlm.evaluation.performance_audit import audit_taskgraph_performance


def test_audit_distinguishes_development_smoke_from_submission(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    run_dir = tmp_path / "complete-system"
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_taskgraph.py",
            "--config",
            "configs/taskgraph/runtime.fake.yaml",
            "--input",
            "tests/fixtures/taskgraph/evaluate_smoke.jsonl",
            "--output-dir",
            str(run_dir),
            "--contract",
            "configs/eval/evaluation_contract_v1.8_local_complete.yaml",
            "--repeat-runs",
            "2",
        ],
    )
    assert evaluate_taskgraph.main() == 0

    development = audit_taskgraph_performance(run_dir)
    submission = audit_taskgraph_performance(run_dir, submission=True)

    assert development["status"] == "pass_with_warnings"
    assert development["blocker_count"] == 0
    assert submission["status"] == "blocked"
    blocker_codes = {
        item["code"] for item in submission["checks"] if item["status"] == "blocker"
    }
    assert "benchmark.warmup" in blocker_codes
    assert "benchmark.repeats" in blocker_codes
    assert "environment.gpu" in blocker_codes
    assert "system.inventory" in blocker_codes


def test_audit_blocks_missing_required_artifacts(tmp_path: Path) -> None:
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"id": "only-row"}) + "\n", encoding="utf-8"
    )

    report = audit_taskgraph_performance(tmp_path)

    assert report["status"] == "blocked"
    assert report["checks"][0]["code"] == "artifacts.required_files"
