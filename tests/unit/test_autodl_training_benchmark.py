from pathlib import Path

from scripts.training.benchmark_autodl_training import (
    Candidate,
    build_benchmark_config,
    build_child_environment,
    build_recommended_config,
    build_recommended_smoke_config,
    parse_candidates,
    resolve_child_python,
    select_best,
)


def base_config() -> dict:
    return {
        "experiment": {"name": "formal"},
        "training": {
            "num_train_epochs": 2,
            "max_steps": None,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
        },
        "data": {"max_train_samples": None, "max_validation_samples": 1024},
        "runtime": {"mock": False},
    }


def test_parse_candidates_preserves_effective_batch() -> None:
    candidates = parse_candidates("4:4:true,8:2:false,16:1:true")

    assert [candidate.effective_batch_size for candidate in candidates] == [16, 16, 16]
    assert candidates[1].gradient_checkpointing is False


def test_benchmark_overrides_do_not_leak_into_recommended_config() -> None:
    candidate = Candidate(16, 1, True)
    benchmark = build_benchmark_config(
        base_config(),
        candidate,
        max_steps=20,
        max_train_samples=512,
    )
    recommended = build_recommended_config(base_config(), candidate)

    assert benchmark["training"]["max_steps"] == 20
    assert benchmark["data"]["max_train_samples"] == 512
    assert benchmark["evaluation"]["do_eval"] is False
    assert recommended["training"]["max_steps"] is None
    assert recommended["data"]["max_train_samples"] is None
    assert recommended["training"]["per_device_train_batch_size"] == 16


def test_select_best_rejects_oom_and_unsafe_memory() -> None:
    results = [
        {"name": "oom", "status": "oom", "memory_safe": True, "train_samples_per_second": 9},
        {
            "name": "unsafe",
            "status": "success",
            "memory_safe": False,
            "train_samples_per_second": 8,
        },
        {
            "name": "safe",
            "status": "success",
            "memory_safe": True,
            "train_samples_per_second": 7,
        },
    ]

    assert select_best(results)["name"] == "safe"


def test_recommended_smoke_has_enough_samples_for_selected_batch() -> None:
    smoke = {
        "experiment": {"name": "smoke"},
        "training": {"max_steps": 5, "gradient_accumulation_steps": 1},
        "data": {"max_train_samples": 16},
    }

    recommended = build_recommended_smoke_config(smoke, Candidate(16, 1, False))

    assert recommended["training"]["per_device_train_batch_size"] == 16
    assert recommended["training"]["gradient_checkpointing"] is False
    assert recommended["data"]["max_train_samples"] == 80


def test_child_environment_includes_source_tree(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PYTHONPATH", "/existing")

    environment = build_child_environment()

    source_tree = str(Path(__file__).resolve().parents[2] / "src").replace("\\", "/")
    assert source_tree in environment["PYTHONPATH"].replace("\\", "/")
    assert "/existing" in environment["PYTHONPATH"]


def test_explicit_autodl_python_takes_priority(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AUTODL_PYTHON", "/opt/rs-vlm/bin/python")

    assert resolve_child_python() == "/opt/rs-vlm/bin/python"
