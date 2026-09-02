from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from sat_rs_vlm.taskgraph import RuntimeRequest, runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import ScalarInt

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
    image = os.environ["TASKGRAPH_SMOKE_IMAGE"]
    question = "How many ships are there?"
    graph = {
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
        "final": {"sources": ["$n1"], "answer_type": "FREE_FORM"},
    }
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
        assert isinstance(scalar.value, int)
        assert scalar.value >= 0
        provider = runtime.providers.counting
        requests = provider.count_requests
        assert requests
        assert requests[0].entire is True
        assert requests[0].target.category == "ship"
        inner = provider._executor.detector
        assert len(inner.calls) >= 1
        assert inner.calls[0].tile.crop_xyxy
    finally:
        runtime.close()
        assert runtime.providers.counting.closed is True
