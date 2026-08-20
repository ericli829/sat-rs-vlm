from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.evaluation.protocols import load_contract
from sat_rs_vlm.evaluation.records import InputValidationError, PredictionRecord
from sat_rs_vlm.evaluation.runner import evaluate_record

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = load_contract(ROOT / "configs" / "eval" / "evaluation_contract_v1.7.yaml")


def _record(**extra: object) -> PredictionRecord:
    row: dict[str, object] = {
        "id": "levir-1",
        "task_type": "change_detection",
        "prediction": "A new building appeared.",
        "reference": "A new building appeared.",
        "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
    }
    row.update(extra)
    return PredictionRecord.from_mapping(row, line_number=1)


def test_v17_strict_profile_rejects_raw_unjudged_levir_caption() -> None:
    with pytest.raises(InputValidationError, match="Run judge_change_captions.py first"):
        evaluate_record(_record(), CONTRACT, None, strict=True)


def test_v17_accepts_resolved_local_judge_result() -> None:
    evaluated, _warnings = evaluate_record(
        _record(
            prediction_changeflag=1,
            binary_prediction="1",
            binary_prediction_source="local_llm_judge",
        ),
        CONTRACT,
        None,
        strict=True,
    )
    assert evaluated.output["predicted_changeflag"] == 1
    assert evaluated.output["binary_prediction_source"] == "local_llm_judge"
    assert evaluated.output["sample_metrics"]["local_judge_decision_valid"] is True


def test_v17_non_strict_profile_marks_missing_local_judge_unresolved() -> None:
    evaluated, _warnings = evaluate_record(_record(), CONTRACT, None, strict=False)
    assert evaluated.output["predicted_changeflag"] is None
    assert evaluated.output["parse_error"] == "missing_required_local_judge_decision"
    assert evaluated.output["sample_metrics"]["local_judge_decision_valid"] is False
