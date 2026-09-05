from __future__ import annotations

import pytest

from sat_rs_vlm.taskgraph.routing import ExecutionMode, ExecutionModeRouter


@pytest.mark.parametrize(
    "task_category",
    (
        "caption",
        "captioning",
        "vqa",
        "visual_question_answering",
        "attribute",
        "classification",
        "scene_classification",
    ),
)
def test_vrsbench_semantic_tasks_use_direct_vlm(task_category: str) -> None:
    assert (
        ExecutionModeRouter().route("VRSBench", task_category)
        is ExecutionMode.DIRECT_VLM
    )


@pytest.mark.parametrize(
    "task_category",
    (
        "count",
        "counting",
        "grounding",
        "visual_grounding",
        "referring_grounding",
        "detection",
    ),
)
def test_vrsbench_localization_and_counting_use_direct_vlm(task_category: str) -> None:
    # Counting/grounding route fully through the semantic 2B VLM (the LAE
    # detector under-counts small objects on VRSBench visuals).
    assert (
        ExecutionModeRouter().route("VRSBench", task_category)
        is ExecutionMode.DIRECT_VLM
    )


@pytest.mark.parametrize(
    "task_category",
    ("change_caption", "change_vqa", "change_detection", "unknown"),
)
def test_levir_cc_uses_direct_vlm(task_category: str) -> None:
    assert (
        ExecutionModeRouter().route("LEVIR-CC", task_category)
        is ExecutionMode.DIRECT_VLM
    )


@pytest.mark.parametrize(
    ("dataset", "task_category"),
    (
        ("MME_RealWorld_RS", "count"),
        ("MME_RealWorld_RS", "color"),
        ("XLRS_Bench", "Counting/Overall counting"),
        ("XLRS_Bench", "Counting/Regional counting"),
    ),
)
def test_other_benchmarks_keep_taskgraph_routing(
    dataset: str, task_category: str
) -> None:
    assert (
        ExecutionModeRouter().route(dataset, task_category)
        is ExecutionMode.TASKGRAPH_UHR
    )
