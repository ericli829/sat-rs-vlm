import math

from sat_rs_vlm.training.rs_merger_expert import (
    _accumulation_window_size,
    resolve_effective_epoch_plan,
)


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


def test_non_divisible_accumulation_flushes_tail_with_actual_normalization():
    total_batches = 10
    accumulation = 4
    window_sizes = [
        _accumulation_window_size(index, total_batches, accumulation)
        for index in range(1, total_batches + 1)
    ]
    flush_indexes = [
        index
        for index in range(1, total_batches + 1)
        if index % accumulation == 0 or index == total_batches
    ]
    assert window_sizes == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
    assert flush_indexes == [4, 8, 10]
    assert sum(1 / window_sizes[index - 1] for index in range(9, 11)) == 1.0
    assert len(flush_indexes) == math.ceil(total_batches / accumulation)


def test_one_effective_epoch_covers_non_divisible_population_and_step_plan():
    plan = resolve_effective_epoch_plan(
        10,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        target_effective_epochs=1.0,
    )
    flush_indexes = [4, 8, 10]
    covered_microbatches = sum(_accumulation_window_size(index, 10, 4) for index in flush_indexes)
    assert plan.optimizer_steps_per_epoch == 3
    assert plan.resolved_max_steps == 3
    assert covered_microbatches == plan.train_size
