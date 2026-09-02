from __future__ import annotations

import json
from pathlib import Path

from sat_rs_vlm.taskgraph.capability_audit import build_target_capability_coverage
from sat_rs_vlm.taskgraph.evaluation_runner import (
    run_taskgraph_evaluation,
    runtime_request_from_sample,
)
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.providers import PlannerFailedError
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime import RuntimeResult
from sat_rs_vlm.taskgraph.runtime_types import ChoiceResult, ImageRef, Region
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
    assert rows[0]["answer"] is None
    assert rows[0]["answer_judgment"]["status"] == "unavailable"
    assert rows[0]["reasoning_chain"]["failure"]["stage"] == "planner"
    assert summary["failure_count"] == 1
    assert summary["success_count"] == 1
    assert runtime.calls == ["bad", "good"]


def test_runner_writes_answer_and_artifact_paths(tmp_path: Path) -> None:
    runtime = _Runtime()
    output = tmp_path / "predictions.jsonl"
    visual_path = tmp_path / "visual.png"
    runtime_result = RuntimeResult(
        execution_mode=ExecutionMode.TASKGRAPH_UHR,
        output=ChoiceResult(selected_ids=("B",), answer_type="CHOICE_SINGLE", raw_response="B"),
        trace=ExecutionTrace(
            sample_id="good",
            execution_mode=ExecutionMode.TASKGRAPH_UHR.value,
            intermediate_output_paths=[str(visual_path)],
        ),
    )

    runtime.run = lambda _request: runtime_result
    run_taskgraph_evaluation(runtime, [_sample("good")], output)

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["answer"] == "B"
    assert row["prediction"]["answer"] == "B"
    assert row["input_image_paths"] == ["fixture.png"]
    assert row["intermediate_output_paths"] == [str(visual_path)]


def test_runner_records_answer_judgment_and_reasoning_chain(tmp_path: Path) -> None:
    runtime = _Runtime()
    output = tmp_path / "predictions.jsonl"
    sample = {**_sample("judged"), "reference_answer": "(A) 1"}

    run_taskgraph_evaluation(runtime, [sample], output)

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["answer_judgment"]["status"] == "correct"
    assert row["answer_judgment"]["comparison"] == "normalized_choice"
    assert row["reasoning_chain"]["final"]["answer"] == "A"
    assert row["reasoning_chain"]["input_image_paths"] == ["fixture.png"]
    assert row["reasoning_chain"]["planner"]["status"] == "not_used"


def test_input_composer_keeps_persistent_visual_paths(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(source)
    composer = InputComposer(tmp_path / "artifacts")
    try:
        checkpoint = composer.artifact_checkpoint()
        model_input = composer.compose_named(
            {"region": Region(ImageRef(str(source)), (2, 3, 12, 14))},
            question="What is here?",
        )
        paths = composer.artifact_paths_since(checkpoint)
        assert list(paths) == model_input.metadata["visual_paths"]
        assert all(Path(path).is_file() for path in paths)
    finally:
        composer.close()


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
