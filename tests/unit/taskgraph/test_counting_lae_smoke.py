from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from sat_rs_vlm.taskgraph import RuntimeRequest, parse_taskgraph, runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import ScalarInt
from sat_rs_vlm.taskgraph.schema import AnswerType

_REQUIRED_ENV = (
    "LAE_DINO_PYTHON",
    "LAE_DINO_SOURCE_ROOT",
    "LAE_DINO_CONFIG_LAE1M",
    "LAE_DINO_CHECKPOINT_LAE1M",
    "LAE_DINO_BERT_ROOT",
    "TASKGRAPH_SMOKE_IMAGE",
)


def _real_lae_ready() -> str | None:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        return f"REAL LAE SMOKE: NOT RUN (missing {', '.join(missing)})"
    image = Path(os.environ["TASKGRAPH_SMOKE_IMAGE"])
    if not image.is_file():
        return "REAL LAE SMOKE: NOT RUN (TASKGRAPH_SMOKE_IMAGE is not a file)"
    for name in (
        "LAE_DINO_SOURCE_ROOT",
        "LAE_DINO_CONFIG_LAE1M",
        "LAE_DINO_CHECKPOINT_LAE1M",
        "LAE_DINO_BERT_ROOT",
    ):
        if not Path(os.environ[name]).exists():
            return f"REAL LAE SMOKE: NOT RUN ({name} does not exist)"
    worker = Path(os.environ["LAE_DINO_PYTHON"])
    if not worker.exists():
        return "REAL LAE SMOKE: NOT RUN (LAE_DINO_PYTHON does not exist)"
    return None


def _count_smoke_graph(image: str, question: str) -> dict:
    return {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "FREE_FORM",
        "inputs": {"image0": {"type": "image", "uri_or_key": image}},
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {
                    "target": {"category": "ship", "attributes": {}},
                    "entire": True,
                },
            }
        ],
        "final": {
            "sources": ["$n1"],
            "question": "",
            "answer_type": "INTEGER",
        },
    }


def test_real_smoke_graph_schema_parses() -> None:
    parsed = parse_taskgraph(_count_smoke_graph("fixture://image", "How many ships are there?"))
    assert parsed.final.answer_type is AnswerType.INTEGER
    assert parsed.final.question == ""
    assert parsed.nodes[0].params["entire"] is True


def test_real_lae_counting_smoke() -> None:
    skip_reason = _real_lae_ready()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    config_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "taskgraph"
        / "runtime.counting_system.real.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Keep LOCATE on DetectionProvider, but do not boot a second tiled LAE sidecar
    # for a COUNT-only smoke. The on-disk yaml still uses tiled detection.
    config["providers"]["detection"] = {"kind": "fake"}
    image = os.environ["TASKGRAPH_SMOKE_IMAGE"]
    question = "How many ships are there?"
    graph = _count_smoke_graph(image, question)
    runtime = runtime_from_config(config)
    try:
        result = runtime.run(
            RuntimeRequest(
                "lae-count-smoke",
                "XLRS_Bench",
                "count",
                question,
                (image,),
                graph=graph,
                target_category="ship",
            )
        )
        scalar = result.store.get("$n1")
        assert isinstance(scalar, ScalarInt)
        assert isinstance(result.output, ScalarInt)
        assert result.output.value == scalar.value
        assert scalar.value >= 0
        provider = runtime.providers.counting
        requests = provider.count_requests
        assert len(requests) == 1
        assert requests[0].entire is True
        assert requests[0].target.category == "ship"
        inner = provider._executor.detector
        assert len(inner.calls) >= 1
        assert inner.calls[0].tile.crop_xyxy
        detection_prov = scalar.provenance.get("detection") or {}
        assert int(detection_prov.get("detector_calls") or 0) >= 1
        assert detection_prov.get("fusion") is not None
        assert detection_prov.get("entire") is True
        print(
            "REAL LAE SMOKE: "
            f"ScalarInt={scalar.value} detector_calls={len(inner.calls)} "
            f"provider={provider.provider_name}",
            flush=True,
        )
    finally:
        runtime.close()
        assert runtime.providers.counting.closed is True
