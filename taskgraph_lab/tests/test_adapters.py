from __future__ import annotations

from pathlib import Path

from taskgraph_lab.datasets.base import normalize_question_type
from taskgraph_lab.datasets.mme_rs import iter_mme_rs
from taskgraph_lab.datasets.xlrs import iter_xlrs
from taskgraph_lab.taskgraph.enums import QuestionType

FIXTURES = Path(__file__).parent / "fixtures"


def test_xlrs_adapter_fixture_is_text_only() -> None:
    samples = list(iter_xlrs(FIXTURES / "xlrs_fixture.json"))
    assert len(samples) == 2
    assert len(samples[1].inputs) == 2
    serialized = samples[0].model_dump(mode="json")
    assert "answer" not in serialized
    assert "bytes" not in str(serialized)
    assert samples[0].metadata["source_category"] == "Object attribute/BBox color"


def test_mme_adapter_fixture_filters_remote_sensing_and_hides_answer() -> None:
    samples = list(iter_mme_rs(FIXTURES / "mme_fixture.json"))
    assert len(samples) == 1
    assert samples[0].sample_id.startswith("mme_rs_")
    assert "Ground truth" not in str(samples[0].model_dump(mode="json"))
    assert samples[0].inputs["image0"].uri_or_key == "remote_sensing/example.png"


def test_choice_normalization_does_not_guess_single_or_multi_cardinality() -> None:
    choices = ["(A) urban", "(B) rural"]
    assert normalize_question_type(None, choices) is QuestionType.MULTIPLE_CHOICE
    assert (
        normalize_question_type("MULTIPLE_CHOICE_SINGLE", choices) is QuestionType.MULTIPLE_CHOICE
    )
    assert normalize_question_type("MULTIPLE_CHOICE_MULTI", choices) is QuestionType.MULTIPLE_CHOICE
