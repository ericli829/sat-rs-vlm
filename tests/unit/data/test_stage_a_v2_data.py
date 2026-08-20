from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.stage_a_v2 import (
    build_canonical_training_population,
    build_stage2_vrs_levir_dataset,
)
from sat_rs_vlm.utils.jsonl import read_jsonl


def _row(sample_id: str, source: str, task: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [],
        "metadata": {"training_source": source},
    }


def _protected_manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "tiers": {
                    "E1": {"sample_ids": ["vrs-protected"]},
                    "E2": {"sample_ids": ["levir-protected"]},
                    "E3": {"sample_ids": ["eval-only"]},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _build_population(tmp_path: Path, *, vrs_count: int, levir_count: int) -> Path:
    protected = _protected_manifest(tmp_path / "tiers.json")
    vrs = [_row(f"vrs-{index}", "VRSBench", "vqa") for index in range(vrs_count)]
    levir = [_row(f"levir-{index}", "LEVIR-CC", "change_detection") for index in range(levir_count)]
    vrs.append(_row("vrs-protected", "VRSBench", "vqa"))
    levir.append(_row("levir-protected", "LEVIR-CC", "change_detection"))
    report = build_canonical_training_population(
        {"VRSBench": vrs, "LEVIR-CC": levir},
        [_row("validation", "VRSBench", "vqa")],
        output_dir=tmp_path / "population",
        protected_evaluation_manifest=protected,
        seed=42,
        source_inputs={"VRSBench": {}, "LEVIR-CC": {}},
        prompt_profiles={"VRSBench": "formal", "LEVIR-CC": "formal"},
    )
    return Path(report["population_manifest"])


def test_population_excludes_all_protected_tiers_and_records_sha(tmp_path: Path) -> None:
    manifest_path = _build_population(tmp_path, vrs_count=4, levir_count=2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        *read_jsonl(manifest["populations"]["VRSBench"]["path"]),
        *read_jsonl(manifest["populations"]["LEVIR-CC"]["path"]),
    ]

    assert {row["id"] for row in rows}.isdisjoint({"vrs-protected", "levir-protected", "eval-only"})
    assert manifest["protected_eval_ids_count"] == 3
    assert manifest["protected_overlap_removed"] == 2
    assert manifest["protected_eval_overlap_count"] == 0
    assert manifest["populations"]["VRSBench"]["sample_count"] == 4
    assert len(manifest["populations"]["VRSBench"]["sha256"]) == 64


def test_stage2_replays_levir_only_when_unique_population_is_short(tmp_path: Path) -> None:
    population = _build_population(tmp_path, vrs_count=12, levir_count=3)
    output = tmp_path / "stage2" / "train.jsonl"
    manifest_path = tmp_path / "stage2" / "manifest.json"
    manifest = build_stage2_vrs_levir_dataset(
        population,
        output_file=output,
        manifest_file=manifest_path,
        seed=42,
        vrs_per_levir=3,
    )
    rows = list(read_jsonl(output))
    replay = [row for row in rows if row["metadata"].get("stage2_replay")]

    assert manifest["vrs_unique_count"] == 12
    assert manifest["levir_unique_count"] == 3
    assert manifest["levir_replay_count"] == 1
    assert manifest["source_distribution"] == {"VRSBench": 12, "LEVIR-CC": 4}
    assert len(rows) == len({row["id"] for row in rows}) == 16
    assert replay[0]["metadata"]["replay_original_id"].startswith("levir-")


def test_stage2_never_discards_unique_levir_to_force_ratio(tmp_path: Path) -> None:
    population = _build_population(tmp_path, vrs_count=6, levir_count=5)
    manifest = build_stage2_vrs_levir_dataset(
        population,
        output_file=tmp_path / "stage2.jsonl",
        manifest_file=tmp_path / "stage2.json",
        seed=42,
        vrs_per_levir=3,
    )

    assert manifest["levir_target_exposures"] == 2
    assert manifest["levir_unique_count"] == 5
    assert manifest["levir_replay_count"] == 0
    assert manifest["total_exposures"] == 11
