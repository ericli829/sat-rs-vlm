import pytest

from sat_rs_vlm.training.training_plan import resolve_training_plan


def test_effective_epoch_budget_resolves_steps() -> None:
    plan = resolve_training_plan(
        unique_samples=1230,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        world_size=1,
        max_steps=None,
        num_train_epochs=None,
        target_effective_epochs=1.5,
        max_effective_epochs=2.0,
        allow_overtrain=False,
    )
    assert plan.effective_batch_size == 16
    assert plan.steps_per_epoch == 77
    assert plan.resolved_max_steps == 116
    assert plan.expected_effective_epochs == pytest.approx(116 / 77)


def test_explicit_overtraining_budget_is_rejected_unless_allowed() -> None:
    kwargs = dict(
        unique_samples=1230,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        world_size=1,
        max_steps=1000,
        num_train_epochs=None,
        target_effective_epochs=1.5,
        max_effective_epochs=2.0,
    )
    with pytest.raises(ValueError, match="exceeds max_effective_epochs"):
        resolve_training_plan(**kwargs, allow_overtrain=False)
    assert resolve_training_plan(**kwargs, allow_overtrain=True).resolved_max_steps == 1000
