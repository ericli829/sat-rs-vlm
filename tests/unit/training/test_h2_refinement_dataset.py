from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from sat_rs_vlm.training import refinement_dataset as refinement_dataset_module
from sat_rs_vlm.training.config import H2DifficultyMixConfig, H2RefinementConfig
from sat_rs_vlm.training.config import HardAdaptationConfig
from sat_rs_vlm.training.refinement_dataset import (
    build_h2_mining_candidates,
    build_h2_refinement_dataset,
    load_protected_e3,
)
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def _row(sample_id: str, source: str, task: str, index: int) -> dict:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "metadata": {
            "dataset": source,
            "training_source": source,
            "qa_type": f"type-{index % 2}" if task == "vqa" else None,
            "changeflag": index % 2 if task == "change_detection" else None,
            "source_task": task,
        },
    }


def _training_rows() -> list[dict]:
    rows: list[dict] = []
    for task in ("vqa", "detection", "captioning"):
        rows.extend(_row(f"vrs-{task}-{i:03d}", "VRSBench", task, i) for i in range(30))
    rows.extend(
        _row(f"levir-change-{i:03d}", "LEVIR-CC", "change_detection", i)
        for i in range(30)
    )
    return rows


def _config() -> H2RefinementConfig:
    return H2RefinementConfig(
        enabled=True,
        source_checkpoint="replay-adapter",
        mining_target_samples=32,
        target_samples=40,
        source_weights={"VRSBench": 0.75, "LEVIR-CC": 0.25},
        difficulty_mix=H2DifficultyMixConfig(
            regular_representative=0.60,
            medium_hard=0.25,
            core_hard=0.15,
        ),
        seed=42,
    )


def _protected(tmp_path: Path, ids: list[str] | None = None) -> tuple[Path, dict]:
    manifest = tmp_path / "evaluation_tiers_manifest.json"
    sample_ids = ids or ["eval-only"]
    manifest.write_text(
        json.dumps(
            {
                "tier_version": "unified-v2",
                "tiers": {
                    "E3": {
                        "sample_count": len(sample_ids),
                        "sha256": "e3-sha",
                        "sample_ids": sample_ids,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, load_protected_e3(manifest)


def _evaluated(row: dict, rank: int) -> dict:
    task = row["task_type"]
    metrics: dict[str, object]
    parse_ok = True
    reference = "answer"
    if task == "vqa":
        metrics = {"normalized_exact_match": False}
    elif task == "detection":
        iou = 0.2 + (rank % 8) * 0.08
        metrics = {
            "parse_success": True,
            "valid_coordinate": True,
            "label_match": True,
            "iou": iou,
            "normalized_center_distance": 0.2,
        }
        reference = '{"label":"vehicle","bbox":[0,0,0.1,0.1]}'
    elif task == "captioning":
        quality = 0.2 + (rank % 8) * 0.05
        metrics = {
            "rouge_l_f1_approx": quality,
            "chrf_approx": quality,
            "cider_d_single_reference_approx": quality * 10,
        }
    else:
        quality = 0.2 + (rank % 8) * 0.05
        metrics = {
            "rouge_l_f1_approx": quality,
            "chrf_approx": quality,
            "cider_d_single_reference_approx": quality * 10,
        }
    return {
        "id": row["id"],
        "task_type": task,
        "reference": reference,
        "parse_ok": parse_ok,
        "sample_metrics": metrics,
        "metadata": row["metadata"],
    }


def test_candidate_builder_is_deterministic_balanced_and_excludes_e3(tmp_path: Path) -> None:
    rows = _training_rows()
    protected_id = rows[0]["id"]
    _, protected = _protected(tmp_path, [protected_id, "eval-only"])
    source = tmp_path / "train.jsonl"
    write_jsonl(source, rows)
    first_file = tmp_path / "first.jsonl"
    second_file = tmp_path / "second.jsonl"
    first = build_h2_mining_candidates(
        rows,
        protected,
        _config(),
        output_file=first_file,
        manifest_file=tmp_path / "first-manifest.json",
        source_training_file=source,
    )
    second = build_h2_mining_candidates(
        rows,
        protected,
        _config(),
        output_file=second_file,
        manifest_file=tmp_path / "second-manifest.json",
        source_training_file=source,
    )
    selected = list(read_jsonl(first_file))
    ids = [row["id"] for row in selected]

    assert len(selected) == 32
    assert len(set(ids)) == 32
    assert protected_id not in ids
    assert first["distribution"]["source"] == {"LEVIR-CC": 8, "VRSBench": 24}
    assert set(first["distribution"]["task"]) == {
        "vqa",
        "detection",
        "captioning",
        "change_detection",
    }
    assert first["output"]["sha256"] == second["output"]["sha256"]
    assert first["duplicate_check"]["passed"] is True
    assert first["evaluation_leakage_check"]["passed"] is True


def test_final_builder_uses_cell_local_ranking_and_exact_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _training_rows()
    _, protected = _protected(tmp_path)
    source = tmp_path / "train.jsonl"
    write_jsonl(source, rows)
    candidates_file = tmp_path / "candidates.jsonl"
    build_h2_mining_candidates(
        rows,
        protected,
        _config(),
        output_file=candidates_file,
        manifest_file=tmp_path / "candidate-manifest.json",
        source_training_file=source,
    )
    candidates = list(read_jsonl(candidates_file))
    evaluated = [_evaluated(row, index) for index, row in enumerate(candidates)]
    predictions = tmp_path / "evaluated_predictions.jsonl"
    write_jsonl(predictions, evaluated)
    output = tmp_path / "h2"
    shortage_policies: list[str] = []
    original_select = refinement_dataset_module.select_hierarchical_tier

    def capture_shortage_policy(*args, shortage_policy: str, **kwargs):
        shortage_policies.append(shortage_policy)
        return original_select(*args, shortage_policy=shortage_policy, **kwargs)

    monkeypatch.setattr(
        refinement_dataset_module,
        "select_hierarchical_tier",
        capture_shortage_policy,
    )
    first = build_h2_refinement_dataset(
        rows,
        candidates,
        evaluated,
        protected,
        _config(),
        HardAdaptationConfig(),
        source_training_file=source,
        mining_candidates_file=candidates_file,
        prediction_source=predictions,
        output_dir=output,
    )
    combined = list(read_jsonl(output / "h2_train.jsonl"))
    role_counts = Counter(row["metadata"]["h2_data_role"] for row in combined)
    source_counts = Counter(row["metadata"]["dataset"] for row in combined)
    ids_by_role = defaultdict(set)
    for row in combined:
        ids_by_role[row["metadata"]["h2_data_role"]].add(row["id"])

    assert role_counts == {
        "regular_representative": 24,
        "medium_hard": 10,
        "core_hard": 6,
    }
    assert source_counts == {"VRSBench": 30, "LEVIR-CC": 10}
    assert not (ids_by_role["regular_representative"] & ids_by_role["medium_hard"])
    assert not (ids_by_role["regular_representative"] & ids_by_role["core_hard"])
    assert not (ids_by_role["medium_hard"] & ids_by_role["core_hard"])
    assert first["difficulty_mode"] == "source_task_cell_rank"
    assert shortage_policies == ["redistribute"]
    assert first["duplicate_check"]["passed"] is True
    assert first["evaluation_leakage_check"]["passed"] is True
    assert first["output_sha256"]["h2_train"]

    hard = [row for row in combined if row["metadata"]["h2_data_role"] != "regular_representative"]
    by_cell_role: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in hard:
        key = (row["metadata"]["dataset"], row["task_type"])
        by_cell_role[key][row["metadata"]["h2_data_role"]].append(
            float(row["metadata"]["hard_score"])
        )
    for roles in by_cell_role.values():
        if roles["core_hard"] and roles["medium_hard"]:
            assert min(roles["core_hard"]) >= max(roles["medium_hard"])

    core_tasks = {
        row["task_type"]
        for row in hard
        if row["metadata"]["dataset"] == "VRSBench"
        and row["metadata"]["h2_data_role"] == "core_hard"
    }
    # VQA has an absolute score of 1.0 while other tasks use lower continuous scales.
    # Cell-local quotas still retain detection/caption core samples.
    assert "vqa" in core_tasks
    assert core_tasks & {"detection", "captioning"}


def test_final_builder_fails_when_one_cell_has_too_few_evaluated_candidates(
    tmp_path: Path,
) -> None:
    rows = _training_rows()
    _, protected = _protected(tmp_path)
    source = tmp_path / "train.jsonl"
    write_jsonl(source, rows)
    candidates_file = tmp_path / "candidates.jsonl"
    build_h2_mining_candidates(
        rows,
        protected,
        _config(),
        output_file=candidates_file,
        manifest_file=tmp_path / "candidate-manifest.json",
        source_training_file=source,
    )
    candidates = list(read_jsonl(candidates_file))
    evaluated = [_evaluated(row, index) for index, row in enumerate(candidates)]
    evaluated = [row for row in evaluated if row["task_type"] != "captioning"]
    predictions = tmp_path / "evaluated_predictions.jsonl"
    write_jsonl(predictions, evaluated)

    with pytest.raises(ValueError, match="exactly the candidate IDs"):
        build_h2_refinement_dataset(
            rows,
            candidates,
            evaluated,
            protected,
            _config(),
            HardAdaptationConfig(),
            source_training_file=source,
            mining_candidates_file=candidates_file,
            prediction_source=predictions,
            output_dir=tmp_path / "h2",
        )
