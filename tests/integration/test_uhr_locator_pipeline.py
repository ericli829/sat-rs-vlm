from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

from sat_rs_vlm.integrations.locators.config import load_locator_config
from sat_rs_vlm.integrations.locators.registry import create_locator

CONFIG = Path("configs/locator/uhr_hierarchical.yaml")


def _image(path: Path) -> Path:
    Image.new("RGB", (900, 900), color=(32, 64, 96)).save(path)
    return path


def test_mock_end_to_end_locator_emits_auditable_trace(tmp_path: Path) -> None:
    image = _image(tmp_path / "fixture.png")
    config = load_locator_config(CONFIG)
    locator = create_locator("hierarchical", config)
    try:
        result = locator.locate(
            image,
            "How many aircraft are visible in the upper-left part?",
        )
    finally:
        locator.close()
    payload = result.to_dict()
    assert payload["task_spec"]["operation"] == "count"
    assert payload["search_plan"]["route"] == "detector_first"
    assert len(payload["search_trace"]) == 9
    assert any(item["selected"] for item in payload["search_trace"])
    assert payload["region_details"]
    assert payload["provider_provenance"]["detector"]["provider"] == "mock"
    assert payload["provider_provenance"]["retriever"]["provider"] == "mock"
    first = payload["search_trace"][0]
    assert set(first["score_components"]) >= {"detector", "retrieval", "spatial", "fused"}
    assert payload["processed_area_ratio"] > 1.0
    assert payload["inspected_area_ratio"] == payload["processed_area_ratio"]
    assert 0.0 <= payload["selected_union_area_ratio"] <= 1.0
    assert 0.0 <= payload["processed_union_area_ratio"] <= 1.0


def test_global_beam_caps_multi_parent_frontier_per_depth(tmp_path: Path) -> None:
    image = _image(tmp_path / "global-beam.png")
    config = load_locator_config(CONFIG)
    config["search"].update(
        {
            "target_view_size": 1,
            "max_depth": 2,
            "max_regions": 64,
            "cumulative_mass": 1.0,
            "max_beam": 4,
            "max_processed_area_ratio": 100.0,
        }
    )
    locator = create_locator("hierarchical", config)
    try:
        payload = locator.locate(image, "How many aircraft are visible?").to_dict()
    finally:
        locator.close()
    depth_one = [item for item in payload["search_trace"] if item["depth"] == 1]
    depth_two = [item for item in payload["search_trace"] if item["depth"] == 2]
    assert len(depth_one) == 9
    assert sum(item["selected"] for item in depth_one) == 4
    assert len(depth_two) == 4 * 9
    assert sum(item["selected"] for item in depth_two) == 4
    assert {item["depth_selected_count"] for item in depth_two} == {4}
    assert len({item["parent_id"] for item in depth_two}) == 4


def test_locator_cli_outputs_json_crops_and_overlay(tmp_path: Path) -> None:
    image = _image(tmp_path / "fixture.png")
    output = tmp_path / "result.json"
    crops = tmp_path / "crops"
    overlay = tmp_path / "overlay.png"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/locator/run_uhr_locator.py",
            "--config",
            str(CONFIG),
            "--image",
            str(image),
            "--question",
            "Where is the airport in the center?",
            "--output",
            str(output),
            "--export-crops",
            str(crops),
            "--export-debug-overlay",
            str(overlay),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["regions_xyxy"]
    assert payload["search_trace"]
    assert overlay.is_file()
    assert all(Path(path).is_file() for path in payload["exports"]["crops"])


def test_diagnostic_sweep_cli_writes_reviewable_artifacts(tmp_path: Path) -> None:
    image = _image(tmp_path / "diagnostic.png")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "test",
                "max_samples": 1,
                "base_overrides": {"search": {"max_depth": 1}},
                "presets": {"E0": {"description": "test", "overrides": {}}},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "samples.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "split": "fixture",
                "usage": "diagnostic only",
                "samples": [
                    {
                        "id": "fixture",
                        "image": str(image),
                        "question": "Where is the airport?",
                        "reference_answer": "center",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sweep"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/uhr_locator_param_sweep.py",
            "--base-config",
            str(CONFIG),
            "--experiment-config",
            str(experiment),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = output / "E0" / "fixture"
    assert (output / "experiment_manifest.json").is_file()
    assert (output / "summary.csv").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "human_review.csv").is_file()
    assert (artifact / "result.json").is_file()
    assert (artifact / "search_trace.json").is_file()
    assert (artifact / "contact_sheet.png").is_file()
