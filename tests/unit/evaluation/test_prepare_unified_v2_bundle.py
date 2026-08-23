from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.evaluation import prepare_unified_v2_bundle as bundle

from sat_rs_vlm.utils.jsonl import write_jsonl


def _row(sample_id: str, task: str, image: str, answer: str) -> dict:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "How many objects are visible?"},
                ],
            },
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"dataset": "VRSBench", "source_task": "fixture"},
    }


def _config(tmp_path: Path) -> Path:
    data_root = tmp_path / "datasets"
    image = data_root / "VRSBench" / "a.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    source = tmp_path / "eval.jsonl"
    train = tmp_path / "train.jsonl"
    write_jsonl(
        source,
        [
            _row("count", "counting", "VRSBench/a.png", "1"),
            _row("guard", "captioning", "VRSBench/a.png", "scene"),
        ],
    )
    write_jsonl(train, [_row("train", "vqa", "VRSBench/a.png", "yes")])
    config = {
        "schema_version": "2.0",
        "tier_version": "unified-v2",
        "seed": 42,
        "evaluation_scope": {"datasets": ["VRSBench"], "tasks": ["counting", "captioning"]},
        "data": {
            "common_image_root": str(data_root),
            "source_files": [str(source)],
            "train_files": [str(train)],
        },
        "stratification": {
            "dataset_weights": {"VRSBench": 1.0},
            "task_balance": "natural",
            "source_shortage_policy": "redistribute",
        },
        "tiers": {"E1": {"target_samples": 1}, "E2": {"target_samples": 2}, "E3": {"mode": "full"}},
    }
    path = tmp_path / "evaluation_tiers.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_bundle_manifest_hashes_match_actual_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bundle, "EXPECTED_E1_ROWS", 1)
    monkeypatch.setattr(bundle, "EXPECTED_E2_ROWS", 2)
    monkeypatch.setattr(bundle, "EXPECTED_ECOUNT_ROWS", 2)
    monkeypatch.setattr(bundle, "EXPECTED_EXACT_VALID", 1)
    output = tmp_path / "bundle"
    result = bundle.prepare_bundle(_config(tmp_path), output, project_root=tmp_path)
    assert result["tier_version"] == "unified-v2"
    assert (output / "evaluation_bundle_manifest.json").is_file()
    files = {
        "E1": "e1_quick.jsonl",
        "E2": "e2_standard.jsonl",
        "E3": "e3_full.jsonl",
        "E_COUNT_V2": "e_count_v2.jsonl",
    }
    for name, filename in files.items():
        record = result["tiers"][name]
        from sat_rs_vlm.evaluation.tiers import canonical_jsonl_sha256, file_sha256

        assert record["raw_sha256"] == file_sha256(output / filename)
        assert record["canonical_jsonl_sha256"] == canonical_jsonl_sha256(output / filename)


def test_bundle_failure_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("old", encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(bundle, "build_counting_focused_tier_v2", fail)
    with pytest.raises(RuntimeError, match="injected build failure"):
        bundle.prepare_bundle(_config(tmp_path), output, project_root=tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "old"
    assert not (output / "e1_quick.jsonl").exists()


def test_semantic_migration_requires_approval_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bundle, "EXPECTED_E1_ROWS", 1)
    monkeypatch.setattr(bundle, "EXPECTED_E2_ROWS", 2)
    monkeypatch.setattr(bundle, "EXPECTED_ECOUNT_ROWS", 2)
    monkeypatch.setattr(bundle, "EXPECTED_EXACT_VALID", 1)
    tracked = tmp_path / "data" / "evaluation" / "tiers_v2"
    tracked.mkdir(parents=True)
    write_jsonl(
        tracked / "e_count_v2.jsonl",
        [
            _row("count", "counting", "VRSBench/a.png", "1"),
            _row("guard", "captioning", "VRSBench/a.png", "different"),
        ],
    )
    (tracked / "e_count_v2_manifest.json").write_text(
        '{"exact_cardinality_valid_count": 1, "exact_cardinality_valid_sample_ids": ["count"]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="requires manual approval"):
        bundle.prepare_bundle(_config(tmp_path), output, project_root=tmp_path)
    diff = tmp_path / "bundle.migration_diff.json"
    assert diff.is_file()
    result = bundle.prepare_bundle(
        _config(tmp_path),
        output,
        project_root=tmp_path,
        allow_benchmark_migration=True,
    )
    assert result["invariants"]["semantic_benchmark_unchanged"] is False
    assert result["invariants"]["benchmark_migration_approved"] is True
    assert result["benchmark_migration"]["approved"] is True
