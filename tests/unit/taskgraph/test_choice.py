from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.choice import ChoiceRequest, ChoiceResolver
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.providers import (
    ChoiceScoringRequest,
    FakeSemanticVLMProvider,
    VLMRequest,
    VLMResult,
)
from sat_rs_vlm.taskgraph.runtime import runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import (
    ChoiceScoreResult,
    ImageRef,
    Label,
    ScalarInt,
    runtime_summary,
)
from sat_rs_vlm.taskgraph.schema import AnswerType

IMAGE = str(Path("tests/fixtures/miniature_dataset/images/vqa.ppm").resolve())
OPTIONS = ("(A) lake", "(B) farm", "(C) mall", "(D) residential")


def test_structured_authoritative_value_maps_without_model_call(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider(choice_scores={"final_choice": {"A": 10.0}})
    resolver = ChoiceResolver(provider, InputComposer(tmp_path / "structured"))

    result = resolver.resolve(
        ChoiceRequest(
            (ScalarInt(7),),
            "Which option matches the count?",
            ("A 5", "B 7", "C 9"),
        )
    )

    assert result.selected_ids == ("B",)
    assert result.choice_id == "B"
    assert result.answer_type == "CHOICE_SINGLE"
    assert result.provenance["method"] == "structured_exact_option_mapping"
    assert provider.choice_calls == []
    assert provider.calls == []
    assert resolver.last_model_input.visual_inputs == ()


def test_out_of_range_count_maps_to_closest_numeric_option_without_model(
    tmp_path: Path,
) -> None:
    provider = FakeSemanticVLMProvider(choice_scores={"final_choice": {"A": 10.0}})
    resolver = ChoiceResolver(provider, InputComposer(tmp_path / "closest"))

    result = resolver.resolve(
        ChoiceRequest(
            (ScalarInt(153),),
            "How many vehicles are there?",
            ("(A) 9", "(B) 1", "(C) 2", "(D) 3", "(E) This image does not feature the count."),
        )
    )

    assert result.selected_ids == ("A",)
    assert result.choice_id == "A"
    assert result.answer_type == "CHOICE_SINGLE"
    assert result.provenance["method"] == "structured_closest_numeric_mapping"
    assert result.provenance["numeric_evidence"] == 153
    # the non-numeric E option must never win when numeric evidence exists
    assert result.selected_ids[0] != "E"
    assert provider.choice_calls == []
    assert provider.calls == []


def test_count_near_option_range_maps_to_closest_and_never_e(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider(choice_scores={"final_choice": {"E": 100.0}})
    resolver = ChoiceResolver(provider, InputComposer(tmp_path / "closest2"))

    result = resolver.resolve(
        ChoiceRequest(
            (ScalarInt(2),),
            "How many count?",
            ("(A) One", "(B) Two", "(C) Five", "(D) Eight", "(E) It does not feature the count."),
        )
    )

    assert result.selected_ids == ("B",)
    assert result.provenance["method"] == "structured_closest_numeric_mapping"
    assert provider.choice_calls == []


def test_non_feature_option_removed_when_forbidden(tmp_path: Path) -> None:
    from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig

    options = (
        "(A) lake",
        "(B) farm",
        "(C) mall",
        "(D) residential",
        "(E) This image doesn't feature the color.",
    )
    # semantic scores rank E highest; with the flag the option must vanish
    provider = FakeSemanticVLMProvider(
        choice_scores={"final_choice": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 100.0}}
    )
    resolver = ChoiceResolver(
        provider,
        InputComposer(tmp_path / "noe"),
        ChoiceSystemConfig(forbid_non_feature_options=True),
    )
    result = resolver.resolve(ChoiceRequest((ImageRef(IMAGE),), "Choose.", options))
    assert result.selected_ids == ("D",)
    # the E option must never be scored
    assert "E" not in result.provenance["scores"]
    assert result.choice_id == "D"

    # without the flag E wins (old behavior)
    provider2 = FakeSemanticVLMProvider(
        choice_scores={"final_choice": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 100.0}}
    )
    resolver2 = ChoiceResolver(provider2, InputComposer(tmp_path / "e"))
    result2 = resolver2.resolve(ChoiceRequest((ImageRef(IMAGE),), "Choose.", options))
    assert result2.selected_ids == ("E",)


def test_reasoning_letters_never_determine_single_choice(tmp_path: Path) -> None:
    provider = FakeSemanticVLMProvider(
        {"final_choice_reasoning": "A looks possible; B is weak; therefore C appears plausible."},
        choice_scores={"final_choice": {"A": -2.0, "B": 4.0, "C": 1.0}},
    )
    resolver = ChoiceResolver(provider, InputComposer(tmp_path / "reasoning"))

    result = resolver.resolve(ChoiceRequest((ImageRef(IMAGE),), "Choose.", OPTIONS[:3]))

    assert result.selected_ids == ("B",)
    assert result.choice_id == "B"
    assert result.answer_type == "CHOICE_SINGLE"
    assert result.raw_response.startswith("A looks possible")
    assert result.provenance["cache_reused"] is True
    assert result.provenance["method"] == "fake_kv_cached_logits"
    assert provider.calls == []
    assert len(provider.choice_calls) == 1


class OptionAwareFake(FakeSemanticVLMProvider):
    def __init__(self, semantic_scores: dict[str, float]) -> None:
        super().__init__()
        self.semantic_scores = semantic_scores

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult:
        base = super().reason_and_choose(request)
        scores = {
            choice_id: self.semantic_scores[option.split()[-1].casefold()]
            for choice_id, option in zip(request.choice_ids, request.option_texts, strict=True)
        }
        if request.answer_type == "CHOICE_SINGLE":
            selected = (max(request.choice_ids, key=lambda item: scores[item]),)
        else:
            selected = tuple(
                choice_id
                for choice_id in request.choice_ids
                if scores[choice_id] > request.multi_select_threshold
            )
        return replace(base, scores=scores, selected_ids=selected)


def test_single_choice_tracks_option_semantics_after_reordering(tmp_path: Path) -> None:
    provider = OptionAwareFake({"lake": 0.1, "farm": 0.9, "mall": 0.2})
    resolver = ChoiceResolver(provider, InputComposer(tmp_path / "reorder"))

    first = resolver.resolve(
        ChoiceRequest((ImageRef(IMAGE),), "Land use?", ("A lake", "B farm", "C mall"))
    )
    second = resolver.resolve(
        ChoiceRequest((ImageRef(IMAGE),), "Land use?", ("A mall", "B lake", "C farm"))
    )

    assert first.selected_ids == ("B",)
    assert second.selected_ids == ("C",)


def test_multi_choice_supports_many_and_canonicalizes_original_order(
    tmp_path: Path,
) -> None:
    provider = OptionAwareFake({"lake": 0.8, "farm": -0.2, "mall": 0.6})
    resolver = ChoiceResolver(
        provider,
        InputComposer(tmp_path / "multi"),
        ChoiceSystemConfig(multi_select_threshold=0.5),
    )

    result = resolver.resolve(
        ChoiceRequest(
            (ImageRef(IMAGE),),
            "Select all.",
            ("A mall", "B farm", "C lake"),
            AnswerType.CHOICE_MULTI,
        )
    )

    assert result.selected_ids == ("A", "C")
    assert result.choice_id is None
    assert result.answer_type == "CHOICE_MULTI"
    assert result.provenance["method"] == "fake_kv_cached_binary_verification"
    scoring_request = provider.choice_calls[-1]
    assert scoring_request.single_choice_suffix.endswith("Final choice:")
    assert "Final choice:" not in scoring_request.multi_verify_template
    assert "Candidate option {choice_id}: {option_text}" in scoring_request.multi_verify_template

    reordered = resolver.resolve(
        ChoiceRequest(
            (ImageRef(IMAGE),),
            "Select all.",
            ("A lake", "B mall", "C farm"),
            AnswerType.CHOICE_MULTI,
        )
    )
    assert reordered.selected_ids == ("A", "B")


def test_multi_choice_with_one_selected_remains_multi(tmp_path: Path) -> None:
    provider = OptionAwareFake({"lake": 0.8, "farm": -0.2, "mall": 0.6})
    result = ChoiceResolver(
        provider,
        InputComposer(tmp_path / "multi-one"),
        ChoiceSystemConfig(multi_select_threshold=0.7),
    ).resolve(
        ChoiceRequest(
            (ImageRef(IMAGE),),
            "Select all.",
            ("A mall", "B farm", "C lake"),
            AnswerType.CHOICE_MULTI,
        )
    )

    assert result.selected_ids == ("C",)
    assert result.choice_id is None
    assert result.answer_type == "CHOICE_MULTI"
    assert runtime_summary(result)["choice_id"] is None
    assert runtime_summary(result)["answer_type"] == "CHOICE_MULTI"


def test_multi_choice_empty_policy_never_secretly_argmaxes(tmp_path: Path) -> None:
    provider = OptionAwareFake({"lake": -0.1, "farm": -0.2})
    resolver = ChoiceResolver(
        provider,
        InputComposer(tmp_path / "empty"),
        ChoiceSystemConfig(multi_select_threshold=0.0, multi_empty_policy="UNRESOLVED"),
    )

    result = resolver.resolve(
        ChoiceRequest(
            (ImageRef(IMAGE),),
            "Select all.",
            ("A lake", "B farm"),
            AnswerType.CHOICE_MULTI,
        )
    )

    assert result.selected_ids == ()
    assert result.choice_id is None
    assert result.answer_type == "CHOICE_MULTI"
    assert result.provenance["empty_multi_status"] == "UNRESOLVED"


def test_precomputed_cached_decision_is_reused_without_second_provider_call(
    tmp_path: Path,
) -> None:
    provider = FakeSemanticVLMProvider()
    score = ChoiceScoreResult(
        selected_ids=("C",),
        scores={"A": -2.0, "B": -1.0, "C": 3.0},
        answer_type="CHOICE_SINGLE",
        reasoning_text="route reasoning",
        provider="qwen3_vl:route_4b",
        model_id="qwen-4b",
        method="kv_cached_single_token_logits",
        cache_reused=True,
    )

    result = ChoiceResolver(provider, InputComposer(tmp_path / "precomputed")).resolve(
        ChoiceRequest((score,), "Route?", ("A north", "B west", "C east"))
    )

    assert result.selected_ids == ("C",)
    assert result.choice_id == "C"
    assert result.answer_type == "CHOICE_SINGLE"
    assert result.raw_response == "route reasoning"
    assert provider.choice_calls == []
    assert provider.calls == []


class InferenceOnlyProvider:
    provider_name = "legacy"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def infer(self, request: VLMRequest) -> VLMResult:
        self.calls += 1
        return VLMResult(self.response, self.provider_name)

    def close(self) -> None:
        return None


def test_legacy_parser_is_explicitly_disabled_by_default(tmp_path: Path) -> None:
    provider = InferenceOnlyProvider("A ... therefore C")
    resolver = ChoiceResolver(provider, InputComposer(tmp_path / "disabled"))
    with pytest.raises(RuntimeError, match="does not implement cached"):
        resolver.resolve(ChoiceRequest((Label("unknown"),), "Choose.", OPTIONS))
    assert provider.calls == 0

    enabled = ChoiceResolver(
        InferenceOnlyProvider("C"),
        InputComposer(tmp_path / "enabled"),
        ChoiceSystemConfig(legacy_regex_fallback=True),
    )
    result = enabled.resolve(ChoiceRequest((Label("unknown"),), "Choose.", OPTIONS))
    assert result.selected_ids == ("C",)
    assert result.answer_type == "CHOICE_SINGLE"
    assert result.provenance["method"] == "legacy_exact_text_parser"


def test_runtime_choice_reuses_semantic_2b_provider_by_default() -> None:
    runtime = runtime_from_config(
        {
            "providers": {
                "detection": {"kind": "fake"},
                "semantic_2b": {"kind": "fake"},
                "route_4b": {"kind": "fake"},
                "choice": {"reuse": "semantic_2b"},
                "region_retriever": {"kind": "fake"},
            }
        }
    )
    try:
        assert runtime.providers.choice is runtime.providers.semantic_2b
    finally:
        runtime.close()
