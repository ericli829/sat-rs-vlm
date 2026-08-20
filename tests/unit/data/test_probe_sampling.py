from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sat_rs_vlm.data.cyclic_training import sha256_file
from sat_rs_vlm.data.probe_sampling import build_balanced_probe_dataset
from sat_rs_vlm.data.stage_a_v2 import BUILDER_VERSION
from sat_rs_vlm.utils.jsonl import write_jsonl


def _row(sample_id: str, source: str, task: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [],
        "metadata": {"training_source": source},
    }


def _population_manifest(tmp_path: Path) -> Path:
    protected = tmp_path / "tiers.json"
    protected.write_text(
        json.dumps({"tiers": {"E1": {"sample_ids": ["eval-1"]}, "E2": {}, "E3": {}}}),
        encoding="utf-8",
    )
    files: dict[str, dict[str, Any]] = {}
    source_rows = {
        "VRSBench": [
            *[_row(f"cap-{index}", "VRSBench", "captioning") for index in range(5)],
            *[_row(f"det-{index}", "VRSBench", "detection") for index in range(5)],
        ],
        "LEVIR-CC": [_row(f"change-{index}", "LEVIR-CC", "change_detection") for index in range(5)],
    }
    for source, rows in source_rows.items():
        path = tmp_path / f"{source}.jsonl"
        write_jsonl(path, rows)
        files[source] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sample_count": len(rows),
        }
    manifest = tmp_path / "population.json"
    manifest.write_text(
        json.dumps(
            {
                "builder_version": BUILDER_VERSION,
                "populations": files,
                "protected": {"manifest": str(protected), "final_overlap": 0},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_balanced_probe_exact_quota_is_deterministic(tmp_path: Path) -> None:
    population = _population_manifest(tmp_path)
    targets = {
        "VRSBench": {"captioning": 2, "detection": 2},
        "LEVIR-CC": {"change_detection": 2},
    }
    first = build_balanced_probe_dataset(
        population,
        targets=targets,
        output_dir=tmp_path / "one",
        seed=42,
        total_samples=6,
    )
    second = build_balanced_probe_dataset(
        population,
        targets=targets,
        output_dir=tmp_path / "two",
        seed=42,
        total_samples=6,
    )
    changed = build_balanced_probe_dataset(
        population,
        targets=targets,
        output_dir=tmp_path / "three",
        seed=43,
        total_samples=6,
    )

    assert first["selected_distribution"] == targets
    assert first["quota_satisfied"] is True
    assert first["output_sha256"] == second["output_sha256"]
    assert first["sample_ids"] != changed["sample_ids"]
    assert first["protected_eval_overlap_count"] == 0


def test_balanced_probe_shortfall_fails_unless_redistribution_is_explicit(
    tmp_path: Path,
) -> None:
    population = _population_manifest(tmp_path)
    targets = {"VRSBench": {"scene_classification": 3}}
    with pytest.raises(ValueError, match=r"requested=3 available=0 shortfall=3"):
        build_balanced_probe_dataset(
            population,
            targets=targets,
            output_dir=tmp_path / "strict",
            total_samples=3,
        )

    report = build_balanced_probe_dataset(
        population,
        targets=targets,
        output_dir=tmp_path / "redistributed",
        total_samples=3,
        quota_shortfall_policy="redistribute",
    )
    assert report["selected_total"] == 3
    assert report["quota_satisfied"] is False
    assert report["redistribution"]["selected"] == 3
    assert report["shortfall"][0]["shortfall"] == 3


def test_probe_dry_run_does_not_write_training_assets(tmp_path: Path) -> None:
    report = build_balanced_probe_dataset(
        _population_manifest(tmp_path),
        targets={"VRSBench": {"captioning": 1}},
        output_dir=tmp_path / "dry",
        total_samples=1,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["output_sha256"] is None
    assert not (tmp_path / "dry").exists()


def test_probe_rejects_a_legacy_round_disguised_as_population(tmp_path: Path) -> None:
    population = _population_manifest(tmp_path)
    payload = json.loads(population.read_text(encoding="utf-8"))
    payload.pop("builder_version")
    population.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not a canonical Stage-A v2 asset"):
        build_balanced_probe_dataset(
            population,
            targets={"VRSBench": {"captioning": 1}},
            output_dir=tmp_path / "rejected",
            total_samples=1,
        )
