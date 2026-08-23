from __future__ import annotations

from sat_rs_vlm.evaluation.change_judge import (
    build_judge_messages,
    conservative_rule_decision,
    parse_judge_output,
)
from sat_rs_vlm.evaluation.parsers import (
    CONTEXTUAL_CHANGE_PARSER_VERSION,
    LEGACY_CHANGE_PARSER_VERSION,
    parse_change_prediction,
    parse_change_prediction_contextual_v3,
    parse_explicit_change_prediction,
)
from sat_rs_vlm.evaluation.records import read_prediction_jsonl


def test_legacy_parser_semantics_remain_exact_and_default_change() -> None:
    assert parse_change_prediction("No change has occurred.").value == 0
    # This compound caption was historically a positive change under the
    # exact-expression parser and must not be silently reinterpreted.
    assert parse_change_prediction("No building changed, but a road appeared.").value == 1
    assert parse_change_prediction("No change").value == 1
    assert LEGACY_CHANGE_PARSER_VERSION == "legacy_exact_no_change_v1"


def test_contextual_v3_handles_global_no_change_and_positive_evidence() -> None:
    no_change = parse_change_prediction_contextual_v3(
        "The two scenes are similar; no change is observed."
    )
    assert no_change.value == 0
    assert no_change.match_type == "contextual_no_change"

    compound = parse_change_prediction_contextual_v3(
        "No buildings changed, but a new road appeared."
    )
    assert compound.value == 1
    assert compound.match_type == "default_change"

    assert parse_change_prediction_contextual_v3("No change").value == 0
    assert parse_change_prediction_contextual_v3('{"changeflag": 1}').match_type == (
        "structured_binary"
    )
    assert CONTEXTUAL_CHANGE_PARSER_VERSION == "levir_contextual_no_change_v3"


def test_semantic_rules_cover_permanent_and_temporary_wording() -> None:
    for caption in (
        "A building was constructed.",
        "Several houses were demolished.",
        "The appearance of a new structure is visible.",
    ):
        decision = conservative_rule_decision(caption)
        assert decision is not None and decision.value == 1
    for caption in (
        "Only a vehicle appeared.",
        "Only lighting and shadows changed.",
    ):
        decision = conservative_rule_decision(caption)
        assert decision is not None and decision.value == 0


def test_explicit_parser_does_not_infer_from_caption_words() -> None:
    assert parse_explicit_change_prediction("1").value == 1
    assert parse_explicit_change_prediction('{"changed": false}').value == 0
    unresolved = parse_explicit_change_prediction("A new road appeared.")
    assert unresolved.value is None
    assert unresolved.match_type == "unresolved"


def test_local_judge_output_is_strict_binary_or_uncertain() -> None:
    assert parse_judge_output("0").value == 0
    assert parse_judge_output("U").value is None
    assert parse_judge_output("The answer is 1").value is None


def test_local_judge_prompt_contains_caption_only() -> None:
    messages = build_judge_messages("a new road appeared")
    assert len(messages) == 2
    assert "a new road appeared" in messages[1]["content"]
    assert "reference" not in messages[1]["content"].lower()
    assert "changeflag" not in messages[1]["content"].lower()


def test_prediction_jsonl_bom_is_accepted(tmp_path) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf"
        + b'{"id":"bom","task_type":"captioning","prediction":"ok",'
        + b'"reference":"ok","metadata":{}}\n'
    )
    records, errors = read_prediction_jsonl(path, strict=True)
    assert errors == []
    assert [record.id for record in records] == ["bom"]
