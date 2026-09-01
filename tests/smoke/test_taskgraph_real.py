from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from sat_rs_vlm.taskgraph.choice import ChoiceRequest
from sat_rs_vlm.taskgraph.providers import ChoiceScoringRequest
from sat_rs_vlm.taskgraph.runtime import runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region, RouteContext, ScalarInt


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
        if kind == "qwen_2b":
            providers["route_4b"] = {"kind": "fake", "default": "A"}
            providers["choice"] = {"reuse": "semantic_2b"}
        elif kind == "qwen_4b":
            providers["semantic_2b"] = {"kind": "fake", "default": "A"}
            providers["choice"] = {"reuse": "semantic_2b"}
    return config


def _local_model(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value or not Path(value).is_dir():
        pytest.skip(f"{variable} is missing or is not a local model directory")
    return Path(value)


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
    _local_model("QWEN3VL_2B_MODEL_DIR")
    runtime = runtime_from_config(_config("qwen_2b"))
    try:
        assert runtime.providers.choice is runtime.providers.semantic_2b
        visual = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ImageRef(image),),
                "Which option best describes this crop?",
                ("A urban", "B water"),
            )
        )
        assert visual.choice_id in {"A", "B"}
        assert visual.provenance["cache_reused"] is True
        assert visual.provenance["method"].startswith("kv_cached_")
        structured = runtime.choice_resolver.resolve(
            ChoiceRequest(
                (ScalarInt(7),),
                "Which option matches this count?",
                ("A 5", "B 7"),
            )
        )
        assert structured.choice_id == "B"
        assert structured.provenance["method"] == "structured_exact_option_mapping"
        assert runtime.choice_resolver.last_model_input.visual_inputs == ()
    finally:
        runtime.close()


@pytest.mark.real_model
def test_real_qwen_route_uses_same_4b_cached_choice() -> None:
    if os.environ.get("RUN_REAL_QWEN") != "1":
        pytest.skip("set RUN_REAL_QWEN=1 for the Qwen smoke")
    image_path = os.environ.get("TASKGRAPH_SMOKE_IMAGE")
    if not image_path or not Path(image_path).is_file():
        pytest.skip("TASKGRAPH_SMOKE_IMAGE is missing")
    _local_model("QWEN3VL_4B_MODEL_DIR")
    with Image.open(image_path) as source:
        width, height = source.size
    image = ImageRef(image_path, width=width, height=height)
    start = Region(image, (0, 0, max(1, width * 0.1), max(1, height * 0.1)))
    goal = Region(
        image,
        (max(0, width * 0.8), max(0, height * 0.8), width, height),
    )
    context = RouteContext(image, start, goal, Region(image, (0, 0, width, height)))
    options = ("A Move east then south", "B Move west then north")
    runtime = runtime_from_config(_config("qwen_4b"))
    try:
        model_input = runtime.composer.compose_named(
            {"context": context},
            question="Which route moves from START to GOAL?",
            options=options,
        )
        result = runtime.providers.route_4b.reason_and_choose(
            ChoiceScoringRequest(
                model_input=model_input,
                answer_type="CHOICE_SINGLE",
                choice_ids=("A", "B"),
                option_texts=options,
                final_suffix=runtime.choice_config.final_suffix,
                purpose="route_choice",
            )
        )
        assert result.selected_ids[0] in {"A", "B"}
        assert result.cache_reused is True
        assert result.provider.endswith(":route_4b")
        assert runtime.providers.choice is runtime.providers.semantic_2b
        assert runtime.providers.choice is not runtime.providers.route_4b
    finally:
        runtime.close()
