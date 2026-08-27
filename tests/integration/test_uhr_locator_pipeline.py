from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
    assert payload["inspected_area_ratio"] > 1.0


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
