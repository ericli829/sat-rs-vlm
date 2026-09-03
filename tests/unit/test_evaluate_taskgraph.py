from __future__ import annotations

import json
from pathlib import Path

import scripts.evaluate_taskgraph as evaluate_taskgraph


def test_complete_system_entrypoint_writes_runtime_and_path_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(project_root)
    output_dir = tmp_path / "complete-system"
    input_file = project_root / "tests/fixtures/taskgraph/evaluate_smoke.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_taskgraph.py",
            "--config",
            "configs/taskgraph/runtime.fake.yaml",
            "--input",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--contract",
            "configs/eval/evaluation_contract_v1.8_local_complete.yaml",
            "--repeat-runs",
            "2",
        ],
    )

    assert evaluate_taskgraph.main() == 0

    prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "system_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (output_dir / "evaluation_metadata.json").read_text(encoding="utf-8")
    )

    assert prediction["telemetry"]["success"] is True
    assert prediction["telemetry"]["route"] == "TASKGRAPH_UHR"
    assert prediction["telemetry"]["timing_ms"]["e2e"] >= 0
    assert len(prediction["telemetry"]["repeat_measurements"]) == 2
    assert prediction["telemetry"]["repeat_output_consistent"] is True
    assert prediction["telemetry"]["vision_input"]["tile_count"] == 1
    assert prediction["telemetry"]["prompt_provenance"]["profile"] == "fixture"
    assert manifest["paths"]["typical"]["sample_count"] == 1
    assert manifest["prompt"]["unique_prompt_hash_count"] == 1
    assert manifest["configuration"]["runtime_config_sha256"]
    assert manifest["performance"]["e2e_ms"]["samples"] == 1
    assert manifest["benchmark"]["repeat_output_policy"].startswith("first_repeat")
    assert metadata["failed_samples"] == 0
