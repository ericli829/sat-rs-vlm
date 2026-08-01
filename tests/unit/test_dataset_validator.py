import json
import shutil
from pathlib import Path

from sat_rs_vlm.data.manifest import validate_dataset

FIXTURE = Path(__file__).parents[1] / "fixtures/miniature_dataset"


def test_miniature_dataset_passes_validation() -> None:
    report = validate_dataset(FIXTURE)
    assert report.valid, report.errors
    assert report.sample_counts["train"] == 3
    assert {"detection", "counting", "vqa"} <= set(report.task_distribution)


def test_duplicate_and_invalid_bbox_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    shutil.copytree(FIXTURE, root)
    rows = (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
    bad = json.loads(rows[0])
    bad["id"] = json.loads(rows[1])["id"]
    bad["boxes"] = [[0.8, 0.1, 0.2, 0.5]]
    rows.append(json.dumps(bad, ensure_ascii=False))
    (root / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = validate_dataset(root)
    assert not report.valid
    assert any("duplicate id" in error for error in report.errors)
    assert any("xyxy" in error for error in report.errors)
