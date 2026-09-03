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
    assert prediction["telemetry"]["vision_input"]["tile_count"] == 0
    assert prediction["telemetry"]["prompt_provenance"]["profile"] == "fixture"
    assert manifest["paths"]["typical"]["sample_count"] == 1
    assert manifest["prompt"]["unique_prompt_hash_count"] == 1
    assert manifest["configuration"]["runtime_config_sha256"]
    assert manifest["performance"]["e2e_ms"]["samples"] == 1
    assert manifest["benchmark"]["repeat_output_policy"].startswith("first_repeat")
    assert metadata["failed_samples"] == 0


def test_complete_system_serializes_direct_grounding_in_evaluator_format(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "grounding-system"
    input_file = tmp_path / "grounding.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "grounding-demo",
                "dataset": "VRSBench",
                "task_type": "detection",
                "target_category": "ship",
                "question": "Find the ship.",
                "images": ["tests/fixtures/miniature_dataset/images/counting.ppm"],
                "reference": json.dumps({"label": "ship", "bbox": [0.0, 0.0, 1.0, 1.0]}),
                "metadata": {
                    "dataset": "VRSBench",
                    "source_task": "referring",
                    "bbox_target_format": "normalized_0_1",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)
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
        ],
    )

    assert evaluate_taskgraph.main() == 0
    prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    parsed = json.loads(prediction["prediction"])
    assert parsed["label"] == "ship"
    assert parsed["bbox"] == [0.0, 0.0, 0.25, 0.25]
    assert prediction["telemetry"]["activated_models"] == ["fake_lae"]
    assert prediction["telemetry"]["vision_input"]["tile_count"] == 0
