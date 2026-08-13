from __future__ import annotations

import math

import pytest

from sat_rs_vlm.training.config import MultitaskLossConfig
from sat_rs_vlm.training.losses import compute_multitask_loss

torch = pytest.importorskip("torch")


def _logits_for_target_probabilities(probabilities: list[list[float]]) -> tuple[object, object]:
    """构造二分类 logits；每行首位为无监督 prompt，后续概率对应 causal labels。"""

    batch_size = len(probabilities)
    sequence_length = max(len(row) for row in probabilities) + 1
    logits = torch.zeros((batch_size, sequence_length, 2), dtype=torch.float32)
    labels = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
    for sample_index, row in enumerate(probabilities):
        for token_index, probability in enumerate(row):
            logits[sample_index, token_index] = torch.tensor(
                [0.0, math.log(probability / (1.0 - probability))]
            )
            labels[sample_index, token_index + 1] = 1
    return logits, labels


def test_token_mean_matches_historical_cross_entropy() -> None:
    logits, labels = _logits_for_target_probabilities([[0.8, 0.7, 0.6], [0.3]])
    result = compute_multitask_loss(
        logits,
        labels,
        ["captioning", "counting"],
        MultitaskLossConfig(mode="token_mean"),
        torch=torch,
    )
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 2),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    assert result.loss.item() == pytest.approx(expected.item())


def test_task_weighted_removes_length_bias_and_token_mean_does_not() -> None:
    caption_tokens = [0.5] * 100
    counting_tokens = [0.5] * 2
    logits, labels = _logits_for_target_probabilities([caption_tokens, counting_tokens])
    weighted = compute_multitask_loss(
        logits,
        labels,
        ["captioning", "counting"],
        MultitaskLossConfig(mode="task_weighted"),
        torch=torch,
    )
    token_mean = compute_multitask_loss(
        logits,
        labels,
        ["captioning", "counting"],
        MultitaskLossConfig(mode="token_mean"),
        torch=torch,
    )

    assert weighted.diagnostics["by_task"]["captioning"]["mean_sample_loss"] == pytest.approx(
        weighted.diagnostics["by_task"]["counting"]["mean_sample_loss"]
    )
    assert token_mean.diagnostics["by_task"]["captioning"]["supervised_tokens"] == 100
    assert token_mean.diagnostics["by_task"]["counting"]["supervised_tokens"] == 2
    assert token_mean.diagnostics["by_task"]["captioning"][
        "effective_loss_numerator_share"
    ] == pytest.approx(100 / 102)
    assert weighted.diagnostics["by_task"]["captioning"][
        "effective_loss_numerator_share"
    ] == pytest.approx(0.5)
    assert weighted.diagnostics["by_task"]["counting"][
        "effective_loss_numerator_share"
    ] == pytest.approx(0.5)


def test_task_weights_follow_public_formula() -> None:
    logits, labels = _logits_for_target_probabilities([[0.8], [0.2]])
    config = MultitaskLossConfig(
        mode="task_weighted",
        task_weights={"captioning": 1.0, "counting": 2.0},
    )
    result = compute_multitask_loss(
        logits,
        labels,
        ["captioning", "counting"],
        config,
        torch=torch,
    )
    expected = (-math.log(0.8) + 2.0 * -math.log(0.2)) / 3.0

    assert result.loss.item() == pytest.approx(expected)


def test_causal_shift_and_ignore_index_are_exact() -> None:
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]]])
    labels = torch.tensor([[-100, 0, -100]])
    result = compute_multitask_loss(
        logits,
        labels,
        ["vqa"],
        MultitaskLossConfig(mode="token_mean"),
        torch=torch,
    )
    expected = torch.nn.functional.cross_entropy(logits[:, 0, :], torch.tensor([0]))

    assert result.loss.item() == pytest.approx(expected.item())
    assert result.diagnostics["by_task"]["vqa"]["supervised_tokens"] == 1


def test_unknown_task_weight_and_missing_metadata_modes() -> None:
    logits, labels = _logits_for_target_probabilities([[0.8], [0.2]])
    config = MultitaskLossConfig(
        mode="task_weighted",
        task_weights={"vqa": 1.0},
        unknown_task_weight=3.0,
        strict_task_metadata=False,
    )
    unknown = compute_multitask_loss(
        logits,
        labels,
        ["vqa", "new_task"],
        config,
        torch=torch,
    )
    expected = (-math.log(0.8) + 3.0 * -math.log(0.2)) / 4.0

    assert unknown.loss.item() == pytest.approx(expected)
    missing = compute_multitask_loss(logits, labels, None, config, torch=torch)
    assert torch.isfinite(missing.loss)
    with pytest.raises(ValueError, match="missing task_types"):
        compute_multitask_loss(
            logits,
            labels,
            None,
            MultitaskLossConfig(strict_task_metadata=True),
            torch=torch,
        )


def test_empty_supervision_is_rejected_after_causal_shift() -> None:
    logits = torch.zeros((1, 3, 2))
    labels = torch.full((1, 3), -100)

    with pytest.raises(ValueError, match="No supervised assistant tokens"):
        compute_multitask_loss(
            logits,
            labels,
            ["vqa"],
            MultitaskLossConfig(),
            torch=torch,
        )


@pytest.mark.parametrize("mode", ["token_mean", "task_weighted"])
def test_loss_preserves_gradient_for_low_precision_compatible_path(mode: str) -> None:
    logits = torch.zeros((2, 4, 3), dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 1], [-100, 2, -100, -100]])

    result = compute_multitask_loss(
        logits,
        labels,
        ["captioning", "counting"],
        MultitaskLossConfig(mode=mode),
        torch=torch,
    )
    result.loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
