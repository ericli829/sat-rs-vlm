from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from sat_rs_vlm.evaluation.change_judge import (
    JudgeDecision,
    parse_judge_output,
    run_local_change_judge,
    run_server_rule_router,
)
from sat_rs_vlm.evaluation.protocols import load_contract
from sat_rs_vlm.evaluation.records import PredictionRecord
from sat_rs_vlm.evaluation.runner import evaluate_record, run_evaluation

ROOT = Path(__file__).resolve().parents[3]
SERVER_CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.8_server_rule_only.yaml"
LOCAL_CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.8_local_complete.yaml"


class FakeJudge:
    model_id = "fake-local-judge"
    model_revision = "test-revision"

    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.captions: list[str] = []

    def judge(self, captions: Sequence[str]) -> list[JudgeDecision]:
        self.captions.extend(captions)
        return [parse_judge_output(output, latency_ms=1.0) for output in self.outputs]


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "id": "no-change",
            "task_type": "change_detection",
            "prediction": "The two scenes remain unchanged.",
            "reference": "No change has occurred.",
            "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
        },
        {
            "id": "positive",
            "task_type": "change_detection",
            "prediction": "A new road appeared beside the buildings.",
            "reference": "A road appeared.",
            "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
        },
        {
            "id": "unresolved",
            "task_type": "change_detection",
            "prediction": "The area looks noticeably different.",
            "reference": "A building was constructed.",
            "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_server_rule_router_is_dependency_free_and_marks_unresolved(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions)

    outputs = run_server_rule_router(predictions, tmp_path / "server-routed")
    rows = _rows(outputs["rule_routed_predictions"])

    assert [row["prediction_changeflag"] for row in rows] == [0, 1, None]
    assert [row["binary_prediction_source"] for row in rows] == [
        "server_semantic_rule",
        "server_semantic_positive_rule",
        "server_rule_unresolved",
    ]
    summary = json.loads(outputs["rule_routing_summary"].read_text(encoding="utf-8"))
    assert summary["resolved_coverage"] == pytest.approx(2 / 3)
    assert (
        json.loads(outputs["rule_routing_manifest"].read_text(encoding="utf-8"))[
            "language_model_loaded"
        ]
        is False
    )


def test_server_partial_evaluation_and_local_completion_are_separate(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions)
    routed = run_server_rule_router(predictions, tmp_path / "server-routed")

    partial = run_evaluation(
        routed["rule_routed_predictions"],
        tmp_path / "server-evaluated",
        contract_path=SERVER_CONTRACT,
        strict=True,
        protected_repository=tmp_path / "protected",
        semantic_enabled=False,
    )
    partial_metrics = json.loads(partial["summary"].read_text(encoding="utf-8"))["by_task"][
        "change_detection"
    ]["metrics"]
    assert partial_metrics["binary_decision_coverage"]["value"] == pytest.approx(2 / 3)
    assert partial_metrics["binary_accuracy"]["status"] == "partial_coverage"
    assert partial_metrics["binary_accuracy"]["num_samples"] == 2

    backend = FakeJudge(["1"])
    completed = run_local_change_judge(
        routed["rule_routed_predictions"],
        tmp_path / "local-completed",
        backend,
        only_unresolved=True,
    )
    assert backend.captions == ["The area looks noticeably different."]
    complete = run_evaluation(
        completed["judged_predictions"],
        tmp_path / "complete-evaluated",
        contract_path=LOCAL_CONTRACT,
        strict=True,
        protected_repository=tmp_path / "protected",
        semantic_enabled=False,
    )
    complete_metrics = json.loads(complete["summary"].read_text(encoding="utf-8"))["by_task"][
        "change_detection"
    ]["metrics"]
    assert complete_metrics["binary_decision_coverage"]["value"] == 1.0
    assert complete_metrics["binary_accuracy"]["status"] == "ok"


def test_local_complete_contract_keeps_auditable_u_out_of_binary_denominator() -> None:
    record = PredictionRecord.from_mapping(
        {
            "id": "uncertain-local",
            "task_type": "change_detection",
            "prediction": "The evidence is unclear.",
            "reference": "No change has occurred.",
            "prediction_changeflag": None,
            "binary_prediction": "U",
            "binary_prediction_source": "local_llm_judge_uncertain",
            "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
        },
        line_number=1,
    )

    evaluated, _warnings = evaluate_record(
        record,
        load_contract(LOCAL_CONTRACT),
        None,
        strict=True,
    )

    assert evaluated.output["predicted_changeflag"] is None
    assert evaluated.output["parse_error"] == "local_judge_unresolved"
