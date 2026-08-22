from sat_rs_vlm.training.rs_merger_expert import resolve_effective_epoch_plan


def test_effective_epoch_plan_tracks_population_size():
    plan = resolve_effective_epoch_plan(
        6374,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        target_effective_epochs=1.0,
    )
    assert plan.effective_batch == 16
    assert plan.optimizer_steps_per_epoch == 399
    assert plan.resolved_max_steps == 399
    assert plan.expected_effective_epochs == 1.0


def test_explicit_max_steps_reports_effective_epochs():
    plan = resolve_effective_epoch_plan(
        100,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=2,
    )
    assert plan.optimizer_steps_per_epoch == 7
    assert plan.expected_effective_epochs == 2 / 7
