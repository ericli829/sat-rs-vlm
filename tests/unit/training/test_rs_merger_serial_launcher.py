from scripts.training.run_rs_merger_experiments import (
    DEFAULT_CONFIGS,
    build_experiment_command,
)


def test_default_merger_matrix_uses_continuations_and_wide_control():
    assert [config.split("/")[-1] for config in DEFAULT_CONFIGS] == [
        "rs_count_merger_c2_cont_4090.yaml",
        "rs_count_merger_c3_cont_4090.yaml",
        "rs_count_merger_c4_wide_4090.yaml",
    ]


def test_launcher_builds_one_child_command_per_variant():
    command = build_experiment_command(
        python="python",
        config=DEFAULT_CONFIGS[0],
        output_root="out",
        max_train_samples=4,
        max_steps=2,
    )
    assert command[:2] == ["python", "scripts/training/train_rs_merger_expert.py"]
    assert command.count("--config") == 1
    assert command[command.index("--max-steps") + 1] == "2"
