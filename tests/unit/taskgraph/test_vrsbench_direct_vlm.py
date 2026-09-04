"""Validate the VRSBench config-driven routing change end to end."""

from __future__ import annotations

from sat_rs_vlm.taskgraph.routing import (
    DatasetExecutionPolicy,
    ExecutionMode,
    ExecutionModeRouter,
)


def test_runtime_example_config_routes_vrsbench_counting_to_direct_vlm() -> None:
    # Load the exact production example config policy section.
    from pathlib import Path

    import yaml

    config_path = (
        Path(__file__).resolve().parents[3]
        / "configs/taskgraph/runtime.real.example.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = DatasetExecutionPolicy.from_mapping(payload.get("dataset_policy"))
    router = ExecutionModeRouter(policy)
    for task in (
        "count",
        "counting",
        "grounding",
        "visual_grounding",
        "referring_grounding",
        "detection",
    ):
        assert router.route("VRSBench", task) is ExecutionMode.DIRECT_VLM, task


def test_default_policy_also_routes_vrsbench_counting_to_direct_vlm() -> None:
    router = ExecutionModeRouter()
    for task in ("count", "counting", "grounding", "visual_grounding"):
        assert router.route("VRSBench", task) is ExecutionMode.DIRECT_VLM, task


def test_other_benchmarks_still_use_taskgraph() -> None:
    router = ExecutionModeRouter()
    assert router.route("MME_RealWorld_RS", "count") is ExecutionMode.TASKGRAPH_UHR
    assert router.route("XLRS_Bench", "Counting/Overall counting") is ExecutionMode.TASKGRAPH_UHR
