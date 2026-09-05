from __future__ import annotations

import torch

from sat_rs_vlm.training.count_aware_loss import (
    auxiliary_ramp,
    categorical_count_loss,
    gaussian_soft_targets,
    inverse_sqrt_class_weights,
    negative_binomial_count_loss,
    normalized_cdf_l1,
    prompt_anchor_indices,
)


def test_prompt_anchor_supports_left_and_right_padding_without_answer_leakage():
    right = torch.tensor([[-100, -100, -100, 7, 8, -100], [-100, -100, 9, 10, -100, -100]])
    right_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    assert prompt_anchor_indices(right, right_mask).tolist() == [2, 1]

    left = torch.tensor([[-100, -100, -100, -100, 7, 8], [-100, -100, -100, 9, 10, 11]])
    left_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    assert prompt_anchor_indices(left, left_mask).tolist() == [3, 2]


def test_soft_targets_are_normalized_and_keep_dominant_exact_class():
    targets = gaussian_soft_targets(torch.tensor([0, 7, 15]), num_classes=16)
    assert torch.allclose(targets.sum(-1), torch.ones(3))
    assert targets.argmax(-1).tolist() == [0, 7, 15]


def test_ordinal_wasserstein_penalizes_7_to_15_more_than_7_to_8():
    near = torch.nn.functional.one_hot(torch.tensor([8]), num_classes=16).float()
    far = torch.nn.functional.one_hot(torch.tensor([15]), num_classes=16).float()
    target = torch.tensor([7])
    assert normalized_cdf_l1(near, target).item() < normalized_cdf_l1(far, target).item()


def test_auxiliary_mask_excludes_invalid_references_and_ramp_is_explicit():
    logits = torch.zeros(3, 16, requires_grad=True)
    weights = inverse_sqrt_class_weights([10] * 16)
    result = categorical_count_loss(
        logits,
        torch.tensor([7, -100, 15]),
        max_count=15,
        class_weights=weights,
        auxiliary_weight=auxiliary_ramp(0.05),
    )
    assert result.valid_count == 2
    assert result.auxiliary_weight == 0.5
    assert torch.isfinite(result.total)
    result.total.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[1]).item() == 0


def test_inverse_sqrt_weights_are_mean_normalized_then_clamped():
    weights = inverse_sqrt_class_weights([1, 4, 16, 0])
    assert weights.min().item() >= 0.5
    assert weights.max().item() <= 2.0


def test_negative_binomial_ablation_is_finite_when_explicitly_selected():
    parameters = torch.zeros(2, 2, requires_grad=True)
    result = negative_binomial_count_loss(
        parameters,
        torch.tensor([2, 7]),
        max_count=15,
    )
    assert result.valid_count == 2
    assert torch.isfinite(result.total)
    result.total.backward()
    assert parameters.grad is not None
