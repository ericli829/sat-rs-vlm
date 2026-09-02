from __future__ import annotations

from pathlib import Path

from sat_rs_vlm.taskgraph import (
    AnswerabilityConfig,
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyStatus,
    fake_runtime,
)
from sat_rs_vlm.taskgraph.answerability import EvidenceSufficiencyExecutor
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.providers import FakeSemanticVLMProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region

IMAGE = str(Path("tests/fixtures/miniature_dataset/images/vqa.ppm").resolve())


def _request() -> EvidenceSufficiencyRequest:
    image = ImageRef(IMAGE)
    return EvidenceSufficiencyRequest(
        "What is the selected building color?",
        Region(image, (0, 0, 8, 8)),
        task_hint="attribute",
        sample_id="answerability-1",
        evidence_version="crop-v2",
    )


def test_answerability_reuses_existing_finite_cache_and_hides_reasoning(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider(
        {"answerability_reasoning": "private free reasoning"},
        choice_scores={
            "answerability": {
                "SUFFICIENT": -1.0,
                "NEED_MORE_EVIDENCE": 4.0,
                "UNRESOLVED": 0.0,
            }
        },
    )
    composer = InputComposer(tmp_path / "answerability")
    try:
        result = EvidenceSufficiencyExecutor(provider, composer).assess(_request())
    finally:
        composer.close()

    assert result.status is EvidenceSufficiencyStatus.NEED_MORE_EVIDENCE
    assert result.cache_reused is True
    assert result.method == "fake_kv_cached_logits"
    assert result.metadata["reasoning_exposed"] is False
    assert "reasoning_text" not in result.metadata
    assert result.metadata["cache_scope"] == "request"
    assert result.metadata["evidence_version"] == "crop-v2"
    assert len(result.metadata["evidence_fingerprint"]) == 64
    assert len(provider.semantic_calls) == 1
    assert provider.semantic_calls[0].purpose == "answerability"
    assert provider.calls == []


def test_answerability_disabled_is_structured_and_does_not_call_model(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider()
    composer = InputComposer(tmp_path / "disabled")
    try:
        result = EvidenceSufficiencyExecutor(
            provider,
            composer,
            AnswerabilityConfig(enabled=False),
        ).assess(_request())
    finally:
        composer.close()
    assert result.status is EvidenceSufficiencyStatus.UNRESOLVED
    assert result.reason_code == "answerability_disabled"
    assert provider.semantic_calls == []


def test_runtime_exposes_answerability_without_inserting_graph_control_flow() -> None:
    runtime = fake_runtime(semantic_choice_scores={"answerability": {"SUFFICIENT": 3.0}})
    try:
        before = len(runtime.graph_executor.router.bindings)
        result = runtime.assess_answerability(_request())
        after = len(runtime.graph_executor.router.bindings)
    finally:
        runtime.close()
    assert result.status is EvidenceSufficiencyStatus.SUFFICIENT
    assert before == after
    assert "ANSWERABILITY" not in {
        operator.value for operator in runtime.graph_executor.router.bindings
    }
