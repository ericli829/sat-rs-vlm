from __future__ import annotations

import pytest

from sat_rs_vlm.training.loss_diagnostics import (
    analyze_causal_lm_batch_loss,
    summarize_multitask_loss_bias,
)

torch = pytest.importorskip("torch")


def test_batch_loss_split_matches_mean_token_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor([[-100, 1, 1, 1], [-100, 0, -100, -100]])
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    expected_loss = torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, 2),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )

    report = analyze_causal_lm_batch_loss(
        logits=logits,
        labels=labels,
        sample_ids=["caption", "vqa"],
        task_types=["captioning", "vqa"],
        torch=torch,
        model_batch_loss=float(expected_loss.item()),
    )

    batch = report["batch_statistics"]
    assert batch["current_batch_loss"] == pytest.approx(batch["manual_batch_token_mean_loss"])
    assert report["task_statistics"]["captioning"]["supervised_tokens"] == 3
    assert report["task_statistics"]["captioning"]["loss_numerator_share"] > 0.5


def test_summary_flags_length_associated_overrepresentation() -> None:
    report = {
        "batch_statistics": {
            "current_batch_loss": 1.0,
            "per_sample_normalized_loss_mean": 1.0,
        },
        "task_statistics": {
            "captioning": {
                "sample_count": 1,
                "supervised_tokens": 9,
                "sum_cross_entropy": 9.0,
                "mean_sample_loss": 1.0,
            },
            "vqa": {
                "sample_count": 1,
                "supervised_tokens": 1,
                "sum_cross_entropy": 1.0,
                "mean_sample_loss": 1.0,
            },
            "counting": {
                "sample_count": 1,
                "supervised_tokens": 1,
                "sum_cross_entropy": 1.0,
                "mean_sample_loss": 1.0,
            },
        },
        "memory": {"peak_memory_allocated_mb": 123.0, "peak_memory_reserved_mb": 456.0},
    }

    summary = summarize_multitask_loss_bias([report], material_share_gap=0.10)

    assert summary["task_statistics"]["captioning"]["length_associated_overrepresentation"]
    assert summary["judgement"]["captioning_or_detection_overrepresented"] == ["captioning"]
    assert summary["judgement"]["vqa_or_counting_underrepresented"] == ["vqa", "counting"]
    assert summary["judgement"]["supports_per_sample_normalized_loss_experiment"]
    assert summary["memory"]["peak_memory_allocated_mb"] == 123.0
