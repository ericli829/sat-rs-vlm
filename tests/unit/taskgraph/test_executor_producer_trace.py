"""Test executor error enrichment (input_producers) via a small GraphExecutor run."""

from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.executor import (
    CapabilityRouter,
    ExecutorBinding,
    GraphExecutor,
    TaskGraphExecutionError,
)
from sat_rs_vlm.taskgraph.operators import GeometryExecutor, OperatorContext, SelectExecutor
from sat_rs_vlm.taskgraph.providers import FakeSemanticVLMProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region, SelectResult, SelectStatus
from sat_rs_vlm.taskgraph.schema import OperatorName, TaskGraph
from sat_rs_vlm.taskgraph.store import RuntimeStore


def _image(path: Path) -> ImageRef:
    return ImageRef(str(path), width=100, height=100)


def test_graph_error_reports_upstream_producer_ops(tmp_path: Path) -> None:
    """SELECT over an unresolved SELECT should name the producer op in details."""

    graph_dict = {
        "version": "taskgraph-v1.1",
        "question": "Which boat?",
        "question_type": "FREE_FORM",
        "choices": None,
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP"},
            },
            {
                # No reference: RELATION is UNRESOLVED ("requires exactly one reference").
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "RELATION", "relation": "RIGHT_OF"},
            },
            {
                # n3 cascades on the UNRESOLVED n2 result.
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n2"},
                "params": {"mode": "ORDINAL", "index": 1, "order": "LEFT_TO_RIGHT"},
            },
        ],
        "final": {"sources": ["$n3"], "question": "Which one?", "answer_type": "TEXT"},
    }
    graph = TaskGraph.model_validate(graph_dict)
    image = _image(tmp_path / "image.png")
    router = CapabilityRouter(
        {
            OperatorName.REGION: ExecutorBinding(GeometryExecutor()),
            OperatorName.SELECT: ExecutorBinding(
                SelectExecutor(FakeSemanticVLMProvider({}))
            ),
        }
    )
    executor = GraphExecutor(router)
    with pytest.raises(TaskGraphExecutionError) as error:
        executor.execute(
            graph,
            RuntimeStore({"$image0": image}),
            sample_id="sample",
            execution_mode="TASKGRAPH_UHR",
            context=OperatorContext("Which one?", (), None),
        )
    details = error.value.details
    assert details["node_id"] == "n3"
    # n3 consumes $n2, produced by SELECT.
    assert details["input_producers"]["candidates"] == "SELECT"


def test_subregion_select_result_materializes_for_count_image() -> None:
    """A SUBREGION SelectResult (single Region) is a valid COUNT.image scope."""
    from sat_rs_vlm.taskgraph.operators import CountExecutor
    from sat_rs_vlm.taskgraph.providers import FakeCountingProvider
    from sat_rs_vlm.taskgraph.runtime_types import ScalarInt
    from sat_rs_vlm.taskgraph.schema import GraphNode

    image = ImageRef("fixture", width=100, height=100)
    region = Region(image, (10, 10, 50, 50), {})
    subregion = SelectResult(region, SelectStatus.OK, "geometry", None, 1.0, {})
    router = CapabilityRouter(
        {
            OperatorName.COUNT: ExecutorBinding(CountExecutor(FakeCountingProvider())),
        }
    )
    count_node = GraphNode.model_validate(
        {
            "id": "n4",
            "op": "COUNT",
            "inputs": {"image": "$n3"},
            "params": {"target": {"category": "car", "attributes": {}}, "entire": False},
        }
    )
    outcome, _ = router.execute(
        count_node,
        {"image": subregion},
        OperatorContext("?", (), None),
    )
    assert isinstance(outcome.value, ScalarInt)
