from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.training.vit_probe import build_probe_dataset, make_checkpoint_evaluable


def _row(sample_id: str, dataset: str, task: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": "请分析图像"},
            {"role": "assistant", "content": "结果"},
        ],
        "task_type": task,
        "metadata": {"dataset": dataset},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_probe_sampling_is_deterministic_and_excludes_all_tiers(tmp_path: Path) -> None:
    rows = [
        _row("eval-only", "VRSBench", "captioning"),
        _row("vrs-cap-1", "VRSBench", "captioning"),
        _row("vrs-det-1", "VRSBench", "detection"),
        _row("vrs-cap-2", "VRSBench", "captioning"),
        _row("vrs-det-2", "VRSBench", "detection"),
        _row("lev-1", "LEVIR-CC", "change_detection"),
        _row("lev-2", "LEVIR-CC", "change_detection"),
        _row("lev-3", "LEVIR-CC", "change_detection"),
    ]
    source = tmp_path / "train.jsonl"
    _write_jsonl(source, rows)
    protected = tmp_path / "tiers.json"
    protected.write_text(
        json.dumps({"tiers": {"E1": {"sample_ids": ["eval-only"]}, "E2": {}, "E3": {}}}),
        encoding="utf-8",
    )
    kwargs = {
        "source_files": [source],
        "protected_evaluation_manifest": protected,
        "target_samples": 4,
        "source_targets": {"VRSBench": 2, "LEVIR-CC": 2},
        "task_targets": {"captioning": 1, "detection": 1, "change_detection": 2},
        "seed": 42,
    }
    first = build_probe_dataset(output_dir=tmp_path / "one", **kwargs)
    second = build_probe_dataset(output_dir=tmp_path / "two", **kwargs)

    assert first["total_samples"] == 4
    assert first["unique_count"] == 4
    assert first["protected_eval_overlap_count"] == 0
    assert first["output_sha256"] == second["output_sha256"]
    assert first["task_distribution"] == second["task_distribution"]
    output_ids = {
        json.loads(line)["id"]
        for line in (tmp_path / "one" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert "eval-only" not in output_ids


def test_probe_checkpoint_promotion_preserves_sidecar(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    checkpoint = experiment / "checkpoint-100"
    (experiment / "processor").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (experiment / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (experiment / "strategy_manifest.json").write_text(
        json.dumps({"strategy": "lora", "adapter_based": True}), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "visual_trainable_weights.safetensors").write_bytes(b"visual")

    make_checkpoint_evaluable(experiment, checkpoint, checkpoint_step=100)

    assert (checkpoint / "strategy_manifest.json").is_file()
    assert (checkpoint / "processor" / "tokenizer.json").is_file()
    assert (checkpoint / "visual_trainable_weights.safetensors").read_bytes() == b"visual"
    manifest = json.loads((checkpoint / "strategy_manifest.json").read_text(encoding="utf-8"))
    assert manifest["probe_checkpoint_step"] == 100


def test_checkpoint_promotion_can_skip_sidecar_only_when_explicitly_requested(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "lora_only_experiment"
    checkpoint = experiment / "checkpoint-10"
    (experiment / "processor").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (experiment / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (experiment / "strategy_manifest.json").write_text(
        json.dumps({"strategy": "lora", "training_stage": "r0"}), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    make_checkpoint_evaluable(
        experiment,
        checkpoint,
        checkpoint_step=10,
        require_visual_sidecar=False,
    )

    assert (checkpoint / "strategy_manifest.json").is_file()
    assert not (checkpoint / "visual_trainable_weights.safetensors").exists()


def test_visual_checkpoint_promotion_still_requires_sidecar_by_default(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "visual_experiment"
    checkpoint = experiment / "checkpoint-10"
    (experiment / "processor").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (experiment / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (experiment / "strategy_manifest.json").write_text(
        json.dumps({"strategy": "lora", "vision_tuning": {"enabled": True}}),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="visual sidecar is missing"):
        make_checkpoint_evaluable(experiment, checkpoint, checkpoint_step=10)
