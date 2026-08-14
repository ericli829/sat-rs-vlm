from __future__ import annotations

import pytest

from sat_rs_vlm.training.config import MultitaskLossConfig
from sat_rs_vlm.training.losses import task_weighted_reference_loss


def test_task_weighted_reference_is_per_sample_not_per_token_or_per_task() -> None:
    # Caption has many more assistant tokens, but the configured objective receives
    # one normalized sample loss per row before applying each row's task weight.
    sample_losses = [2.0, 4.0, 8.0]
    tasks = ["captioning", "counting", "counting"]
    config = MultitaskLossConfig(
        task_weights={"captioning": 1.0, "counting": 2.0},
    )

    actual = task_weighted_reference_loss(sample_losses, tasks, config)

    assert actual == pytest.approx((1.0 * 2.0 + 2.0 * 4.0 + 2.0 * 8.0) / 5.0)
    assert actual != pytest.approx((2.0 + (4.0 + 8.0) / 2.0) / 2.0)


def test_task_weighted_reference_rejects_missing_rows() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        task_weighted_reference_loss([], [], MultitaskLossConfig())
