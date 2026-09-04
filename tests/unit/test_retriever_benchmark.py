from pathlib import Path

import pytest
from PIL import Image
from scripts.retriever_benchmark import _union_area, run_benchmark, sliding_grid_boxes


def test_sliding_grid_boxes_uses_uniform_overlapping_windows() -> None:
    boxes = sliding_grid_boxes(100, 100, 3, 0.5)
    assert len(boxes) == 9
    assert boxes[0] == (0.0, 0.0, 50.0, 50.0)
    assert boxes[4] == (25.0, 25.0, 75.0, 75.0)
    assert boxes[-1] == (50.0, 50.0, 100.0, 100.0)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_sliding_grid_boxes_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="window_ratio"):
        sliding_grid_boxes(100, 100, 3, ratio)


def test_union_area_does_not_double_count_overlap() -> None:
    assert _union_area([(0, 0, 10, 10), (5, 0, 15, 10)]) == 150.0


def test_benchmark_reports_retrieval_and_gate_metrics(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (90, 90), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        f'{{"id":"one","image":"{image.as_posix()}","query":"harbor","gt_boxes":[[0,0,30,30]]}}\n',
        encoding="utf-8",
    )
    report = run_benchmark(manifest, "mock", {}, grid_size=3, top_k=1)
    assert report["schema_version"] == "region-retriever-benchmark-v1"
    assert report["samples"] == 1
    assert "recall_at_k" in report["metrics"]
    assert "recall_at_1" in report["metrics"]
    assert "reciprocal_rank" in report["metrics"]
    assert "random_recall_at_k" in report["metrics"]
    assert "oracle_recall" in report["metrics"]
    assert "gate_recall" in report["metrics"]
    assert report["rows"][0]["scored_regions"] == 9
    assert len(report["rows"][0]["ranked_region_indices"]) == 9
    assert len(report["rows"][0]["region_scores"]) == 9


def test_benchmark_reports_sliding_window_geometry(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        f'{{"id":"one","image":"{image.as_posix()}","query":"ship","gt_boxes":[[20,20,80,80]]}}\n',
        encoding="utf-8",
    )
    report = run_benchmark(
        manifest,
        "mock",
        {},
        grid_size=3,
        top_k=5,
        candidate_window_ratio=0.5,
    )
    assert report["candidate_window_ratio"] == 0.5
    assert report["rows"][0]["candidate_window_ratio"] == 0.5
    assert report["rows"][0]["scored_regions"] == 9
    assert report["rows"][0]["selected_area_ratio"] == pytest.approx(1.25)
    assert report["rows"][0]["processed_area_ratio"] == pytest.approx(1.25)
    assert report["rows"][0]["mean_selected_roi_area_ratio"] == pytest.approx(0.25)
    assert 0.0 < report["rows"][0]["selected_union_area_ratio"] <= 1.0
