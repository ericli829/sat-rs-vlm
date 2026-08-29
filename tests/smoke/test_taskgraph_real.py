from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from sat_rs_vlm.taskgraph.choice import ChoiceRequest
from sat_rs_vlm.taskgraph.runtime import runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, ScalarInt


def _config(kind: str) -> dict:
    path = os.environ.get("TASKGRAPH_REAL_CONFIG")
    if not path:
        pytest.skip("TASKGRAPH_REAL_CONFIG is not configured")
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    providers = config["providers"]
    if kind == "lae":
        for name in ("semantic_2b", "route_4b", "choice"):
            providers[name] = {"kind": "fake", "default": "A"}
        providers["region_retriever"] = {"kind": "fake"}
    else:
        providers["detection"] = {"kind": "fake"}
        providers["region_retriever"] = {"kind": "fake"}
        providers.pop("planner", None)
    return config


@pytest.mark.real_model
def test_real_lae_detection_contract() -> None:
    if os.environ.get("RUN_REAL_LAE") != "1":
        pytest.skip("set RUN_REAL_LAE=1 for the LAE smoke")
    image = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image or not Path(image).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    runtime = runtime_from_config(_config("lae"))
    try:
        from sat_rs_vlm.taskgraph.providers import DetectionRequest
        from sat_rs_vlm.taskgraph.schema import TargetSpec

        result = runtime.providers.detection.detect(
            DetectionRequest(ImageRef(image), TargetSpec(category="ship"))
        )
        for entity in result.detections.entities:
            x0, y0, x1, y1 = entity.region.bbox_xyxy_global
            assert x0 < x1 and y0 < y1
            assert entity.score is None or 0.0 <= entity.score <= 1.0
            assert entity.provenance.get("provider")
    finally:
        runtime.close()


@pytest.mark.real_model
def test_real_qwen_visual_and_structured_choice_contracts() -> None:
    if os.environ.get("RUN_REAL_QWEN") != "1":
        pytest.skip("set RUN_REAL_QWEN=1 for the Qwen smoke")
    image = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image or not Path(image).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    runtime = runtime_from_config(_config("qwen"))
    try:
        visual = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ImageRef(image),),
                "Which option best describes this crop?",
                ("A urban", "B water"),
            )
        )
        assert visual.choice_id in {"A", "B"}
        structured = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ScalarInt(7),),
                "Which option matches this count?",
                ("A 5", "B 7"),
            )
        )
        assert structured.choice_id in {"A", "B"}
        assert runtime.choice_resolver.last_model_input.visual_inputs == ()
    finally:
        runtime.close()
