import json
from pathlib import Path

from sat_rs_vlm.application.reliability_service import ReliabilityExperimentService


def test_mock_report_pipeline_uses_standard_layout(tmp_path: Path) -> None:
    config = {
        "experiment": {
            "name": "pipeline",
            "seed": 42,
            "execution_mode": "smoke_mock",
        },
        "protection": {"strategies": ["no_protection", "output_guard_vote"]},
    }
    service = ReliabilityExperimentService(
        config,
        project_root=Path(__file__).resolve().parents[3],
        output_root=tmp_path,
        command="pytest reliability pipeline",
    )

    layout = service.run(smoke_case="output-guard", run_id="fixed")
    summary = json.loads((layout.metrics / "summary.json").read_text(encoding="utf-8"))
    smoke = json.loads((layout.root / "smoke_report.json").read_text(encoding="utf-8"))

    assert (layout.root / "config_resolved.yaml").is_file()
    assert (layout.faults / "injection_records.jsonl").is_file()
    assert (layout.predictions / "clean_fault_pairs.jsonl").is_file()
    assert (layout.protection / "strategy_results.json").is_file()
    assert summary["execution_mode"] == "smoke_mock"
    assert smoke["mock_results_are_real_model_metrics"] is False


def test_real_mode_never_falls_back_to_mock(tmp_path: Path) -> None:
    service = ReliabilityExperimentService(
        {
            "experiment": {
                "name": "real",
                "seed": 42,
                "execution_mode": "real_inference",
            },
            "model": {"adapter_path": tmp_path / "missing"},
        },
        project_root=Path(__file__).resolve().parents[3],
        output_root=tmp_path,
        command="pytest real",
    )

    try:
        service.run(mode="full", run_id="missing-assets")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("real_inference unexpectedly succeeded or fell back to Mock")
    report = json.loads(
        (tmp_path / "reliability/real/missing-assets/run_report.json").read_text(encoding="utf-8")
    )
    assert report["success"] is False
    assert report["execution_mode"] == "real_inference"
