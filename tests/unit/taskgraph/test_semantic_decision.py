from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.contracts import validate_runtime_output
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import OperatorContext, SemanticExecutor
from sat_rs_vlm.taskgraph.providers import (
    FakeSemanticVLMProvider,
    VLMRequest,
    VLMResult,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    Boolean,
    ChoiceScoreResult,
    Entity,
    ImageRef,
    Label,
    LabelSet,
    Region,
)
from sat_rs_vlm.taskgraph.schema import GraphNode
from sat_rs_vlm.taskgraph.semantic_decision import SemanticDecisionConfig

IMAGE = str(Path("tests/fixtures/miniature_dataset/images/vqa.ppm").resolve())


def _node(op: str, inputs: dict[str, str], params: dict[str, object]) -> GraphNode:
    return GraphNode.model_validate({"id": "n1", "op": op, "inputs": inputs, "params": params})


def _entity() -> Entity:
    image = ImageRef(IMAGE)
    return Entity(Region(image, (0, 0, 8, 8)), "target", 0.9)


def test_relation_uses_highest_canonical_score_and_never_reasoning_text(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider(
        {"semantic_relation_reasoning": "The prose repeatedly claims RIGHT_OF."},
        choice_scores={
            "semantic_relation": {"LEFT_OF": 8.0, "RIGHT_OF": -3.0},
        },
    )
    composer = InputComposer(tmp_path / "relation")
    executor = SemanticExecutor(provider)
    try:
        outcome = executor.execute(
            _node("RELATION", {"subject": "$n0", "reference": "$n2"}, {}),
            {"subject": _entity(), "reference": _entity()},
            OperatorContext("Where is the subject?", (), composer),
        )
    finally:
        composer.close()

    assert isinstance(outcome.value, Label)
    assert outcome.value.value == "LEFT_OF"
    assert outcome.value.provenance["canonical"] is True
    assert outcome.value.provenance["reasoning_text"].startswith("The prose")
    assert outcome.value.provenance["method"] == "kv_cached_categorical"
    assert outcome.value.provenance["decision_metadata"]["visual_prefill_count"] == 1
    assert outcome.value.provenance["decision_metadata"]["session_released"] is True
    assert provider.calls == []
    assert len(provider.semantic_calls) == 1


@pytest.mark.parametrize(
    ("scores", "expected"),
    [({"YES": 3.0, "NO": -1.0}, True), ({"YES": -2.0, "NO": 4.0}, False)],
)
def test_motion_uses_binary_scores_even_when_reasoning_is_uncertain(
    tmp_path: Path,
    scores: dict[str, float],
    expected: bool,
) -> None:
    provider = FakeSemanticVLMProvider(
        {"semantic_motion_reasoning": "uncertain; no parseable yes/no answer here"},
        choice_scores={"semantic_motion": scores},
    )
    composer = InputComposer(tmp_path / f"motion-{expected}")
    try:
        outcome = SemanticExecutor(provider).execute(
            _node("MOTION", {"source": "$n0"}, {}),
            {"source": _entity()},
            OperatorContext("Is it moving?", (), composer),
        )
    finally:
        composer.close()

    assert isinstance(outcome.value, Boolean)
    assert outcome.value.value is expected
    assert outcome.value.provenance["method"] == "kv_cached_binary"
    assert outcome.value.provenance["cache_reused"] is True
    assert provider.calls == []


def test_classify_with_label_space_is_canonical_cached_decision(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider(
        {"semantic_classify_reasoning": "It might be an airport in prose."},
        choice_scores={"semantic_classify": {"airport": -1.0, "harbor": 5.0, "industrial": 0.2}},
    )
    composer = InputComposer(tmp_path / "classify")
    try:
        outcome = SemanticExecutor(provider).execute(
            _node(
                "CLASSIFY",
                {"source": "$n0"},
                {"label_space": ["airport", "harbor", "industrial"]},
            ),
            {"source": _entity()},
            OperatorContext("Classify.", (), composer),
        )
    finally:
        composer.close()

    assert outcome.value.value == "harbor"
    assert outcome.value.provenance["canonical"] is True
    assert outcome.value.provenance["cache_reused"] is True
    assert provider.calls == []


def test_multilabel_scores_each_label_from_one_reasoning_cache(tmp_path: Path) -> None:
    labels = ("airport", "harbor", "residential")
    provider = FakeSemanticVLMProvider(
        choice_scores={
            "semantic_multilabel_classify": {
                "airport": 2.0,
                "harbor": -1.0,
                "residential": 3.0,
            }
        }
    )
    composer = InputComposer(tmp_path / "multilabel")
    try:
        outcome = SemanticExecutor(provider).execute(
            _node(
                "MULTILABEL_CLASSIFY",
                {"source": "$n0"},
                {"label_space": list(labels)},
            ),
            {"source": _entity()},
            OperatorContext("Select all classes.", (), composer),
        )
    finally:
        composer.close()

    assert isinstance(outcome.value, LabelSet)
    assert outcome.value.values == ("airport", "residential")
    metadata = outcome.value.provenance["decision_metadata"]
    assert metadata["reasoning_pass_count"] == 1
    assert metadata["reasoning_cache_mode"] == "fork_per_option"
    assert metadata["choice_scored_tokens"] == 3
    assert len(provider.semantic_calls) == 1


def test_attribute_finite_config_and_open_generation_are_explicit(tmp_path: Path) -> None:
    entity = _entity()
    finite_provider = FakeSemanticVLMProvider(
        choice_scores={"semantic_attribute": {"white": 4.0, "gray": -1.0}}
    )
    open_provider = FakeSemanticVLMProvider({"attribute": "weathered metal"})
    finite_composer = InputComposer(tmp_path / "finite-attribute")
    open_composer = InputComposer(tmp_path / "open-attribute")
    try:
        finite = SemanticExecutor(
            finite_provider,
            semantic_config=SemanticDecisionConfig(attributes={"color": ("white", "gray")}),
        ).execute(
            _node("ATTRIBUTE", {"entity": "$n0"}, {"attribute": "color"}),
            {"entity": entity},
            OperatorContext("Color?", (), finite_composer),
        )
        opened = SemanticExecutor(open_provider).execute(
            _node("ATTRIBUTE", {"entity": "$n0"}, {"attribute": "material"}),
            {"entity": entity},
            OperatorContext("Material?", (), open_composer),
        )
    finally:
        finite_composer.close()
        open_composer.close()

    assert finite.value.value == "white"
    assert finite.value.provenance["canonical"] is True
    assert opened.value.value == "weathered metal"
    assert opened.value.provenance["method"] == "free_text_generation"
    assert opened.value.provenance["canonical"] is False


def test_intermediate_vlm_reason_remains_answer(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider({"vlm_reason": "open visual explanation"})
    composer = InputComposer(tmp_path / "reason")
    try:
        outcome = SemanticExecutor(provider).execute(
            _node("VLM_REASON", {"image": "$image0"}, {"question": "$question"}),
            {"image": ImageRef(IMAGE)},
            OperatorContext("Explain.", (), composer),
        )
    finally:
        composer.close()
    assert isinstance(outcome.value, Answer)
    assert outcome.value.text == "open visual explanation"
    assert outcome.value.provenance["semantic_method"] == "free_generation"


class _InferenceOnlySemanticProvider:
    provider_name = "inference_only"

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, request: VLMRequest) -> VLMResult:
        self.calls += 1
        return VLMResult("RIGHT_OF", self.provider_name)

    def close(self) -> None:
        return None


def test_finite_semantics_do_not_fall_back_to_legacy_text_parser(tmp_path: Path) -> None:
    provider = _InferenceOnlySemanticProvider()
    composer = InputComposer(tmp_path / "no-parser")
    try:
        with pytest.raises(RuntimeError, match="cached finite decisions"):
            SemanticExecutor(provider).execute(  # type: ignore[arg-type]
                _node("RELATION", {"subject": "$n0", "reference": "$n2"}, {}),
                {"subject": _entity(), "reference": _entity()},
                OperatorContext("Relation?", (), composer),
            )
    finally:
        composer.close()
    assert provider.calls == 0


def test_output_contract_is_execution_dependent() -> None:
    validate_runtime_output("ATTRIBUTE", Label("white"), final_choice_fusion=False)
    validate_runtime_output(
        "ATTRIBUTE",
        ChoiceScoreResult(
            selected_ids=("A",),
            scores={"A": 1.0},
            answer_type="CHOICE_SINGLE",
            reasoning_text="reason",
            provider="fake",
            model_id="fake",
            method="cached",
            cache_reused=True,
        ),
        final_choice_fusion=True,
    )
    with pytest.raises(TypeError, match="normal execution"):
        validate_runtime_output(
            "ATTRIBUTE",
            ChoiceScoreResult(
                selected_ids=("A",),
                scores={"A": 1.0},
                answer_type="CHOICE_SINGLE",
                reasoning_text=None,
                provider="fake",
                model_id="fake",
                method="cached",
                cache_reused=True,
            ),
            final_choice_fusion=False,
        )
