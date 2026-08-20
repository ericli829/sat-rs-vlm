"""Unit tests for local-judge LoRA data preparation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "evaluation"
    / "train_local_change_judge_lora.py"
)
SPEC = importlib.util.spec_from_file_location("train_local_change_judge_lora", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "evaluation"


def test_read_examples_and_validation_metrics() -> None:
    examples = MODULE.read_examples(FIXTURES / "local_judge_sft_examples.jsonl")

    assert [(row.sample_id, row.caption, row.target_label) for row in examples] == [
        ("one", "A road was built.", "1"),
        ("two", "No change.", "0"),
    ]
    summary = MODULE.accuracy_summary(
        [
            {"target_label": "1", "prediction": 1},
            {"target_label": "0", "prediction": 1},
            {"target_label": "1", "prediction": None},
            {"target_label": "0", "prediction": 0},
        ]
    )
    assert summary["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert summary["accuracy"] == 0.5
    assert summary["unresolved_rate_on_scored_binary"] == 0.25
    assert summary["num_scored_binary_samples"] == 4


def test_read_examples_rejects_u_target() -> None:
    try:
        MODULE.read_examples(FIXTURES / "local_judge_sft_invalid_u.jsonl")
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected U-only example to be rejected")


def test_caption_content_not_synthetic_audit_id_defines_leakage() -> None:
    development = [MODULE.SftExample("caption-0001", "A building was constructed.", "1", 1.0)]
    holdout = [MODULE.SftExample("caption-0001", "No change occurred.", "0", 1.0)]

    assert {row.sample_id for row in development} & {row.sample_id for row in holdout}
    assert not ({row.caption for row in development} & {row.caption for row in holdout})
