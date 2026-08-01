from pathlib import Path

from sat_rs_vlm.data.manifest import (
    load_dataset_manifest,
    load_manifest_split,
)
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset

FIXTURE = Path(__file__).parents[1] / "fixtures/miniature_dataset"


def test_manifest_split_loads_in_qwen_dataset() -> None:
    manifest = load_dataset_manifest(FIXTURE / "dataset_manifest.json")
    rows = load_manifest_split(FIXTURE, manifest, "smoke")
    dataset = Qwen3VLDataset(FIXTURE / manifest.splits["smoke"])
    assert len(rows) == len(dataset) == 2
    assert dataset[0]["task_type"] == "detection"
