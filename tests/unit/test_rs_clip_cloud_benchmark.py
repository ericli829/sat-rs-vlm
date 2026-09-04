import json
from pathlib import Path

import pytest
from PIL import Image
from scripts.cloud.run_rs_clip_benchmark import (
    aggregate_rows,
    model_provider_config,
    stable_stage_rows,
    validate_images,
    verify_prerequisite,
)


def test_staged_rows_are_deterministic_and_nested() -> None:
    rows = [{"id": f"sample-{index}"} for index in range(100)]
    first = stable_stage_rows(rows, 17, 50)
    repeated = stable_stage_rows(list(reversed(rows)), 17, 50)
    larger = stable_stage_rows(rows, 17, 80)
    assert [row["id"] for row in first] == [row["id"] for row in repeated]
    assert [row["id"] for row in first] == [row["id"] for row in larger[:50]]
    with pytest.raises(ValueError, match="requires 101"):
        stable_stage_rows(rows, 17, 101)


def test_cloud_manifest_validation_rejects_normalized_boxes(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (128, 128), "white").save(image)
    valid = {
        "id": "valid",
        "image": str(image),
        "category": "airport",
        "gt_boxes": [[10, 10, 80, 80]],
    }
    assert validate_images([valid])["rows"] == 1
    invalid = {**valid, "id": "normalized", "gt_boxes": [[0.1, 0.2, 0.9, 1.0]]}
    with pytest.raises(ValueError, match="appears normalized"):
        validate_images([invalid])


def test_cloud_aggregation_excludes_latency_warmup_only() -> None:
    rows = []
    for index, latency in enumerate((1000.0, 100.0, 110.0)):
        row = {
            key: 1.0
            for key in (
                "recall_at_k",
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "reciprocal_rank",
                "average_precision",
                "ndcg_at_k",
                "oracle_recall",
                "random_recall_at_k",
                "gt_positive_region_coverage",
                "mean_gt_coverage",
                "top1_gt_coverage",
                "topk_union_gt_coverage",
                "selected_area_ratio",
                "gate_recall",
                "detector_call_reduction",
            )
        }
        row.update({"id": str(index), "latency_ms": latency, "cache_hits": 0})
        rows.append(row)
    metrics = aggregate_rows(rows, warmup_rows=1)
    assert metrics["recall_at_5"] == 1.0
    assert metrics["steady_latency_ms"]["median"] == 105.0


def test_model_provider_config_validates_artifact_type(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    assert model_provider_config({"checkpoint": str(checkpoint)}, False)["checkpoint"] == str(
        checkpoint
    )
    assert model_provider_config({"model_path": str(model_dir)}, False)["model_path"] == str(
        model_dir
    )
    with pytest.raises(FileNotFoundError, match="checkpoint is not a file"):
        model_provider_config({"checkpoint": str(model_dir)}, False)


def test_prerequisite_requires_complete_five_model_same_manifest(tmp_path: Path) -> None:
    status_dir = tmp_path / "smoke50"
    status_dir.mkdir()
    status_path = status_dir / "tier_status.json"
    with pytest.raises(RuntimeError, match="missing"):
        verify_prerequisite(tmp_path, "standard500", "source-sha")

    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "models": ["remoteclip"],
                "source_manifest_sha256": "source-sha",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="all five models"):
        verify_prerequisite(tmp_path, "smoke50", "source-sha")

    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "models": [
                    "remoteclip",
                    "georsclip",
                    "farslip",
                    "satelliteclip",
                    "git_rsclip",
                ],
                "source_manifest_sha256": "different-sha",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different source manifest"):
        verify_prerequisite(tmp_path, "smoke50", "source-sha")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["source_manifest_sha256"] = "source-sha"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    verify_prerequisite(tmp_path, "smoke50", "source-sha")
