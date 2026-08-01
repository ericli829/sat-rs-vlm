from pathlib import Path

import pytest

from sat_rs_vlm.data.manifest import (
    DatasetManifest,
    load_dataset_manifest,
    resolve_split_path,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/miniature_dataset"


def test_fixture_manifest_is_valid() -> None:
    manifest = load_dataset_manifest(FIXTURE / "dataset_manifest.json")
    assert manifest.dataset_name == "sat-rs-vlm-miniature"
    assert resolve_split_path(FIXTURE, manifest, "smoke") == (FIXTURE / "smoke.jsonl").resolve()


def test_absolute_split_path_is_rejected() -> None:
    payload = {
        "dataset_name": "bad",
        "dataset_version": "1",
        "root_format": "external",
        "image_path_type": "relative",
        "coordinate_format": "xyxy",
        "coordinate_range": [0, 1],
        "splits": {
            "train": "C:/secret/train.jsonl",
            "validation": "v.jsonl",
            "test": "t.jsonl",
            "smoke": "s.jsonl",
        },
    }
    with pytest.raises(ValueError, match="relative"):
        DatasetManifest.model_validate(payload)
