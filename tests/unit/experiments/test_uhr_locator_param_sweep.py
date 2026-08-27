from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from scripts.experiments.uhr_locator_param_sweep import (
    deep_merge,
    load_experiment_config,
    load_sample_manifest,
    render_sample_artifacts,
)

from sat_rs_vlm.integrations.locators.types import LocatorError

EXPERIMENT = Path("configs/experiments/uhr_locator/diagnostic_v1.yaml")
SAMPLES = Path("configs/experiments/uhr_locator/diagnostic_v1_samples.yaml")


def test_diagnostic_experiment_config_is_staged_and_inherits_e0() -> None:
    config = load_experiment_config(EXPERIMENT)
    assert len(config["presets"]) == 16
    assert config["max_samples"] == 5
    baseline = config["base_overrides"]
    assert baseline["detector"]["provider"] == "lae_dino_lae1m"
    assert baseline["retriever"]["provider"] == "visrag"
    halo = deep_merge(baseline, config["presets"]["H0_halo_005"]["overrides"])
    assert halo["search"]["halo_ratio"] == 0.05
    assert halo["search"]["cumulative_mass"] == 0.85
    assert halo["search"]["max_beam"] == 3
    assert config["presets"]["E0_baseline"]["overrides"] == {}


def test_diagnostic_manifest_uses_five_high_resolution_samples() -> None:
    manifest = load_sample_manifest(SAMPLES)
    assert len(manifest["samples"]) == 5
    assert manifest["dataset"] == "MME-RealWorld-RS"
    assert "NOT FOR FINAL HYPERPARAMETER TUNING" in manifest["usage"]
    assert all("MME_REALWORLD_RS_ROOT" in sample["image"] for sample in manifest["samples"])


def test_experiment_config_rejects_invalid_base_overrides(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("base_overrides: []\npresets:\n  E0: {}\n", encoding="utf-8")
    with pytest.raises(LocatorError, match="base_overrides"):
        load_experiment_config(path)


def test_diagnostic_rendering_is_headless(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (320, 240), color=(30, 60, 90)).save(image_path)
    payload = {
        "processed_area_ratio": 1.2,
        "selected_union_area_ratio": 0.25,
        "processed_union_area_ratio": 0.8,
        "depth_reached": 1,
        "regions_xyxy": [[10.0, 20.0, 140.0, 160.0]],
        "scores": [0.9],
        "region_details": [
            {
                "depth": 1,
                "score": 0.9,
                "score_components": {
                    "detector": {"raw": 0.8},
                    "retrieval": {"raw": 0.7},
                },
            }
        ],
        "search_trace": [
            {
                "region_id": "root.0",
                "parent_id": "root",
                "depth": 1,
                "core_xyxy": [10.0, 20.0, 140.0, 160.0],
                "selected": True,
                "fused_score": 0.9,
                "selection_probability": 0.75,
                "stop_reasons": ["target_view_size"],
            },
            {
                "region_id": "root.1",
                "parent_id": "root",
                "depth": 1,
                "core_xyxy": [150.0, 20.0, 300.0, 160.0],
                "selected": False,
                "fused_score": 0.2,
                "selection_probability": 0.25,
                "stop_reasons": [],
            },
        ],
        "provider_provenance": {
            "detector": {
                "metadata": {
                    "boxes_xyxy": [[20.0, 30.0, 50.0, 60.0]],
                    "scores": [0.8],
                    "queries": [],
                }
            }
        },
    }
    output_dir = tmp_path / "artifacts"
    exports = render_sample_artifacts(
        image_path,
        payload,
        {"question": "Where is the airport?", "reference_answer": "upper right"},
        output_dir,
    )
    assert Path(exports["contact_sheet"]).is_file()
    assert Path(exports["detector_proposals"]).is_file()
    assert Path(exports["search_final"]).is_file()
    assert len(exports["depth_overlays"]) == 1
    assert Path(exports["crops"][0]).is_file()
