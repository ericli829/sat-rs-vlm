from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.integrations.vlm_fo1_worker import (
    MockBackend,
    PipelineConfig,
    process_request,
    validate_request,
)

from sat_rs_vlm.evaluation.counting_protocol import PROTOCOL_NAME
from sat_rs_vlm.evaluation.ensemble import (
    EnsembleComparisonError,
    majority_vote_counting,
    median_vote_counting,
    pairwise_counting_comparison,
)
from sat_rs_vlm.integrations.vlm_fo1 import (
    build_counting_prompt,
    extract_count_target_phrase,
    parse_region_indexes,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many small vehicles are visible?", "small vehicles"),
        ("How many large storage tanks are visible?", "large storage tanks"),
        ("How many planes are visible?", "planes"),
        ("How many tennis courts can be seen?", "tennis courts"),
        ("How many ships are there?", "ships"),
    ],
)
def test_target_phrase_extraction_is_open_vocabulary(question: str, expected: str) -> None:
    result = extract_count_target_phrase(question)
    assert result.supported
    assert result.phrase == expected


@pytest.mark.parametrize(
    "question",
    [
        "How many unique object categories are present?",
        "How many lanes are on the highway?",
        "How many objects are visible?",
        "Are there more ships than harbors?",
    ],
)
def test_unsupported_target_is_explicit(question: str) -> None:
    result = extract_count_target_phrase(question)
    assert result.status == "unsupported"
    assert result.phrase is None


def test_prompt_profiles_are_switchable() -> None:
    question = "How many small vehicles are visible?"
    assert build_counting_prompt(question, "small vehicles", "plain") == question
    assert "integer only" in build_counting_prompt(question, "small vehicles", "integer")
    assert '"count"' in build_counting_prompt(question, "small vehicles", "json")
    assert build_counting_prompt(question, "small vehicles", "official_fo1").startswith(
        "How many small vehicles are there in this image?"
    )


def test_region_index_parser_preserves_evidence_and_zero_count() -> None:
    parsed = parse_region_indexes(
        "<ground>ships</ground><objects><region2><region2><region0></objects>",
        proposal_count=3,
    )
    assert parsed["parse_ok"] is True
    assert parsed["selected_region_indexes"] == [2, 0]
    zero = parse_region_indexes("<ground>ships</ground><objects></objects>", proposal_count=3)
    assert zero["parse_ok"] is True
    assert zero["selected_region_indexes"] == []


def test_region_index_parser_rejects_out_of_range() -> None:
    parsed = parse_region_indexes(
        "<ground>ships</ground><objects><region4></objects>", proposal_count=2
    )
    assert parsed["parse_ok"] is False
    assert parsed["invalid_region_indexes"] == [4]


def test_worker_json_protocol_and_failure_handling() -> None:
    config = PipelineConfig(model_path="mock", upn_checkpoint="mock")
    backend = MockBackend()
    request = {
        "id": "sample-1",
        "image": "missing.png",
        "question": "How many small vehicles are visible?",
        "target_phrase": "small vehicles",
    }
    response = process_request(request, backend, config)
    assert response["status"] == "ok"
    assert response["fo1_count"] == 2
    assert response["selected_region_indexes"] == [0, 1]
    bad = process_request({**request, "reference": "2"}, backend, config)
    assert bad["status"] == "failed"
    assert bad["failure_stage"] == "protocol_guard"
    upper_bad = process_request({**request, "Reference": "2"}, backend, config)
    assert upper_bad["failure_stage"] == "protocol_guard"
    mismatch = process_request({**request, "target_phrase": "ships"}, backend, config)
    assert mismatch["failure_stage"] == "protocol_guard"


def test_validate_request_marks_unsupported_without_backend_call() -> None:
    request = {
        "id": "sample-unsupported",
        "image": "image.png",
        "question": "How many lanes are on the highway?",
        "target_phrase": "",
    }
    normalized, response = validate_request(request, prompt_profile="official_fo1")
    assert normalized is None
    assert response is not None
    assert response["status"] == "unsupported"
    assert response["target_status"] == "unsupported"


def _row(sample_id: str, prediction: str, reference: str = "2") -> dict[str, str]:
    return {
        "id": sample_id,
        "task_type": "counting",
        "prediction": prediction,
        "reference": reference,
    }


def test_pairwise_alignment_and_oracle() -> None:
    result = pairwise_counting_comparison(
        [_row("a", "2"), _row("b", "1")],
        [_row("a", "1"), _row("b", "2")],
    )
    assert result["pairwise_prediction_agreement"] == 0.0
    assert result["oracle_accuracy"] == 1.0
    assert result["correctness_overlap"]["a_only_correct"] == 1
    assert result["correctness_overlap"]["b_only_correct"] == 1


def test_duplicate_id_rejected() -> None:
    with pytest.raises(EnsembleComparisonError, match="duplicate"):
        pairwise_counting_comparison(
            [_row("a", "1"), _row("a", "2")], [_row("a", "1"), _row("b", "2")]
        )


def test_majority_vote_has_no_router_search() -> None:
    result = majority_vote_counting([[_row("a", "2")], [_row("a", "2")], [_row("a", "1")]])
    assert result["accuracy"] == 1.0
    assert result["threshold_search"]["performed"] is False


def test_median_vote_preserves_missing_predictions() -> None:
    result = median_vote_counting([[_row("a", "1")], [_row("a", "")], [_row("a", "3")]])
    assert result["rows"][0]["prediction"] == 2
    assert result["threshold_search"]["performed"] is False


def test_mock_evaluator_writes_standard_outputs(tmp_path: Path) -> None:
    from scripts.evaluation.evaluate_vlm_fo1 import evaluate

    source = tmp_path / "tier.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "task_type": "counting",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "image.png"},
                            {"type": "text", "text": "How many ships are visible?"},
                        ],
                    },
                    {"role": "assistant", "content": '{"count":2}'},
                ],
                "metadata": {"dataset": "VRSBench"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    outputs = evaluate(
        {
            "scope": "e_count_v2",
            "input": source,
            "output_dir": output_dir,
            "image_root": None,
            "max_samples": 4,
            "backend": "mock",
            "worker_python": Path(__import__("sys").executable),
            "worker_script": Path("scripts/integrations/vlm_fo1_worker.py"),
            "model": Path("mock"),
            "upn_checkpoint": Path("mock"),
            "device": "cpu",
            "proposal_score_threshold": 0.3,
            "proposal_top_k": 100,
            "nms_threshold": 0.8,
            "max_new_tokens": 4096,
            "temperature": 0.0,
            "top_p": 0.05,
            "prompt_profile": "official_fo1",
            "audit": audit,
        }
    )
    assert set(outputs) == {"predictions", "metrics", "summary", "provenance", "diagnostics"}
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics["metrics_protocol"] == PROTOCOL_NAME
    assert metrics["n"] == 1
    prediction = json.loads(outputs["predictions"].read_text(encoding="utf-8"))
    assert prediction["prediction"] == "2"
    assert {
        "id",
        "task_type",
        "question",
        "reference",
        "prediction",
        "proposal_boxes",
        "proposal_scores",
        "selected_region_indexes",
        "selected_region_boxes",
    } <= prediction.keys()


def test_full_quantity_scope_keeps_parseable_quantity_population() -> None:
    from scripts.evaluation.evaluate_vlm_fo1 import _select_rows

    def row(sample_id: str, reference: str, task_type: str = "counting") -> dict[str, object]:
        return {
            "id": sample_id,
            "task_type": task_type,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "image.png"},
                        {"type": "text", "text": "How many ships are visible?"},
                    ],
                },
                {"role": "assistant", "content": reference},
            ],
            "metadata": {"dataset": "VRSBench", "qa_type": "object quantity"},
        }

    selected = _select_rows(
        [row("numeric", "2"), row("non_numeric", "Multiple"), row("other", "2", "vqa")],
        "full_vrsbench_quantity",
    )
    assert [item["id"] for item in selected] == ["numeric"]
