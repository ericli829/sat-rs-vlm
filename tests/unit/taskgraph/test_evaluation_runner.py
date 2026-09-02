from __future__ import annotations

import json
from pathlib import Path

from sat_rs_vlm.taskgraph.capability_audit import build_target_capability_coverage
from sat_rs_vlm.taskgraph.evaluation_runner import (
    run_taskgraph_evaluation,
    runtime_request_from_sample,
)
from sat_rs_vlm.taskgraph.providers import PlannerFailedError
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime import RuntimeResult
from sat_rs_vlm.taskgraph.runtime_types import ChoiceResult
from sat_rs_vlm.taskgraph.tracing import ExecutionTrace


class _Runtime:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []

    def run(self, request):
        self.calls.append(request.sample_id)
        if request.sample_id in self.failures:
            raise PlannerFailedError("planner_failed")
        return RuntimeResult(
            execution_mode=ExecutionMode.TASKGRAPH_UHR,
            output=ChoiceResult(
                selected_ids=("A",),
                answer_type="CHOICE_SINGLE",
                raw_response="A",
            ),
            trace=ExecutionTrace(
                sample_id=request.sample_id,
                execution_mode=ExecutionMode.TASKGRAPH_UHR.value,
            ),
        )


def _sample(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset": "MME_RealWorld_RS",
        "task_category": "count",
        "question": "How many ships are there?",
        "image_paths": ["fixture.png"],
        "choices": ["(A) 1", "(B) 2"],
        "question_type": "MULTIPLE_CHOICE_SINGLE",
    }


def test_runner_writes_failure_and_continues(tmp_path: Path) -> None:
    runtime = _Runtime({"bad"})
    output = tmp_path / "predictions.jsonl"

    summary = run_taskgraph_evaluation(
        runtime,
        [_sample("bad"), _sample("good")],
        output,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["failure", "success"]
    assert rows[0]["error_type"] == "planner_failed"
    assert rows[0]["stage"] == "planner"
    assert summary["failure_count"] == 1
    assert summary["success_count"] == 1
    assert runtime.calls == ["bad", "good"]


def test_runner_resumes_only_successful_sample_ids(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    first_runtime = _Runtime()
    run_taskgraph_evaluation(first_runtime, [_sample("done")], output)

    second_runtime = _Runtime()
    summary = run_taskgraph_evaluation(
        second_runtime,
        [_sample("done"), _sample("next")],
        output,
    )

    assert second_runtime.calls == ["next"]
    assert summary["skipped_completed_samples"] == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_runner_records_malformed_sample_and_continues(tmp_path: Path) -> None:
    runtime = _Runtime()
    output = tmp_path / "predictions.jsonl"

    summary = run_taskgraph_evaluation(
        runtime,
        [{"question": "missing id"}, _sample("good")],
        output,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["sample_id"] == "<sample-1>"
    assert rows[0]["status"] == "failure"
    assert rows[0]["error_type"] == "ValueError"
    assert rows[1]["sample_id"] == "good"
    assert summary["failure_count"] == 1
    assert runtime.calls == ["good"]


def test_runtime_request_adapter_supports_mme_shape(tmp_path: Path) -> None:
    request = runtime_request_from_sample(
        {
            "Question_id": "mme-1",
            "Question Type": "Multiple Choice",
            "Image": "remote/image.png",
            "Text": "How many cars?",
            "Answer choices": ["(A) 1", "(B) 2"],
            "Category": "count",
        },
        image_root=tmp_path,
    )

    assert request.sample_id == "mme-1"
    assert request.image_paths == (str((tmp_path / "remote/image.png").resolve()),)
    assert request.options == ("(A) 1", "(B) 2")
    assert request.question_type.value == "MULTIPLE_CHOICE_SINGLE"


def test_capability_audit_records_bridge_and_unresolved_targets(tmp_path: Path) -> None:
    source = tmp_path / "planner.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "one",
                "messages": [
                    {"role": "assistant", "content": 'n1=LOCATE($image0,T("bridge"))'},
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "two",
                "messages": [
                    {"role": "assistant", "content": 'n1=LOCATE($image0,T("unknown lake"))'},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_target_capability_coverage(
        [source],
        ontology_path="configs/eval/semantic/remote_sensing_ontology.json",
    )
    categories = {row["target_category"]: row for row in report["categories"]}

    assert categories["bridge"]["capability"] == "DETECTOR"
    assert categories["unknown lake"]["capability"] == "UNRESOLVED"
