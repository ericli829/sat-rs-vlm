from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sat_rs_vlm.integrations.locators.config import load_locator_config
from sat_rs_vlm.integrations.locators.registry import create_locator

CONFIG = Path("configs/locator/uhr_hierarchical.yaml")


def _real_enabled() -> bool:
    return os.environ.get("RUN_UHR_LOCATOR_REAL_SMOKE") == "1"


def _image() -> Path:
    value = os.environ.get("UHR_LOCATOR_SMOKE_IMAGE", "")
    if not value:
        pytest.skip("UHR_LOCATOR_SMOKE_IMAGE is not configured")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.skip(f"UHR locator smoke image is missing: {path}")
    return path


def _run(*, detector: bool, retriever: bool, questions: tuple[str, ...]) -> None:
    if not _real_enabled():
        pytest.skip("set RUN_UHR_LOCATOR_REAL_SMOKE=1 for real provider smoke")
    config = load_locator_config(CONFIG)
    config["detector"].update(
        {"enabled": detector, "provider": "lae_dino_lae1m" if detector else "mock"}
    )
    config["retriever"].update(
        {"enabled": retriever, "provider": "visrag" if retriever else "mock"}
    )
    config["scorers"]["detector"]["enabled"] = detector
    config["scorers"]["retrieval"]["enabled"] = retriever
    locator = create_locator("hierarchical", config)
    try:
        results = [locator.locate(_image(), question) for question in questions]
    finally:
        locator.close()
    summaries = []
    for result in results:
        assert result.regions_xyxy
        assert result.search_trace
        candidates = []
        for item in result.search_trace:
            components = item["score_components"]
            candidates.append(
                {
                    "bbox": item["view_xyxy"],
                    "detector_score": components["detector"]["raw"],
                    "retrieval_score": components["retrieval"]["raw"],
                    "spatial_score": components["spatial"]["raw"],
                    "fused_score": item["fused_score"],
                    "selected": item["selected"],
                }
            )
        summaries.append(
            {
                "question": result.task_spec.raw_question,
                "provider_provenance": result.provider_provenance,
                "latency_ms": result.latency_ms,
                "candidates": candidates,
            }
        )
    print(json.dumps(summaries, ensure_ascii=False))


@pytest.mark.real_model
@pytest.mark.gpu
def test_real_lae_locator_smoke() -> None:
    _run(
        detector=True,
        retriever=False,
        questions=("How many aircraft are visible in the northern part?",),
    )


@pytest.mark.real_model
@pytest.mark.gpu
def test_real_visrag_locator_smoke() -> None:
    _run(
        detector=False,
        retriever=True,
        questions=(
            "Where are the airplanes?",
            "Is the ship north of the harbor?",
            "Why is this area important for transportation?",
        ),
    )


@pytest.mark.real_model
@pytest.mark.gpu
def test_real_lae_visrag_combined_locator_smoke() -> None:
    _run(
        detector=True,
        retriever=True,
        questions=("How many aircraft are visible in the northern part?",),
    )
