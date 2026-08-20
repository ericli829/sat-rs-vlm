from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sat_rs_vlm.training.config import load_training_config
from sat_rs_vlm.training.stage_a_v2 import (
    resolve_stage_epoch_plan,
    training_command,
    validate_stage2_sampler_coverage,
)


ROOT = Path(__file__).parents[3]


def _runner_module():
    path = ROOT / "scripts/training/run_qwen3vl_4b_stage_a_v2.py"
    spec = importlib.util.spec_from_file_location("stage_a_v2_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(sample_id: str, source: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "task_type": "vqa" if source == "VRSBench" else "change_detection",
        "metadata": {"training_source": source},
    }


def test_epoch_plan_derives_half_and_final_steps_from_effective_batch() -> None:
    plan = resolve_stage_epoch_plan(
        101,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
        world_size=1,
    )

    assert plan.effective_batch_size == 16
    assert plan.steps_per_epoch == 7
    assert plan.half_checkpoint_step == 4
    assert plan.final_step == 7


def test_stage_commands_distinguish_adapter_continuation_and_trainer_resume() -> None:
    command = training_command(
        "python",
        config="r1.yaml",
        train_file="train.jsonl",
        validation_file="val.jsonl",
        output_dir="adapter",
        save_steps=10,
        initial_adapter="r0-adapter",
        resume_checkpoint="adapter/checkpoint-5",
    )

    assert command[command.index("--initial-adapter") + 1] == "r0-adapter"
    assert command[command.index("--resume-from-checkpoint") + 1] == ("adapter/checkpoint-5")


def test_stage2_coverage_first_sampler_keeps_every_exposure() -> None:
    rows = [*[_row(f"v-{i}", "VRSBench") for i in range(12)]]
    rows.extend(_row(f"l-{i}", "LEVIR-CC") for i in range(4))
    report = validate_stage2_sampler_coverage(
        rows,
        source_batch_pattern=["VRSBench", "VRSBench", "VRSBench", "LEVIR-CC"],
        batch_size=4,
        seed=42,
    )

    assert report["valid"] is True
    assert report["unique_indices"] == 16
    assert report["duplicate_count"] == 0
    assert report["missing_indices"] == []


def test_formal_r0_and_r1_configs_lock_training_contract(monkeypatch) -> None:
    monkeypatch.setenv("QWEN3VL_4B_MODEL_DIR", "/models/qwen3-vl-4b")
    monkeypatch.setenv("DATA_ROOT", "/datasets")
    monkeypatch.setenv("OUTPUT_ROOT", "/outputs")
    r0 = load_training_config(ROOT / "configs/train/qwen3vl_4b_stage_a_v2_r0_strong_lora_4090.yaml")
    r1 = load_training_config(
        ROOT / "configs/train/qwen3vl_4b_stage_a_v2_r1_visual_reinforce_4090.yaml"
    )

    assert r0.lora.initial_adapter_dir is None
    assert r0.training.learning_rate == 1.0e-4
    assert r0.training.per_device_train_batch_size * r0.training.gradient_accumulation_steps == 16
    assert r0.vision_tuning.enabled is False
    assert r1.lora.initial_adapter_dir == ("/outputs/qwen3vl_4b_stage_a_v2/pending/r0/adapter")
    assert r1.vision_tuning.unfreeze_last_n_blocks == 2
    assert r1.vision_tuning.train_main_merger is True
    assert r1.optimization.lora_lr == 2.0e-5
    assert r1.optimization.visual_merger_lr == 1.0e-5
    assert r1.optimization.vision_lr == 2.0e-6


def test_runner_evaluation_identity_accepts_e2_without_weakening_tier_check(
    tmp_path: Path,
) -> None:
    module = _runner_module()
    root = tmp_path / "evaluation_e2" / "evaluation_v1_5"
    root.mkdir(parents=True)
    (root / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_tier": "E2",
                "evaluation_tier_sha256": "tier-sha",
                "evaluated_samples": 3000,
                "latency_context": {"eval_batch_size": 4},
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(json.dumps({"tasks": {}}), encoding="utf-8")

    identity = module._evaluation_identity(root.parent, expected_tier="E2")
    assert identity["tier"] == "E2"
    assert identity["eval_batch_size"] == 4
    with pytest.raises(ValueError, match="Expected E1 evaluation"):
        module._evaluation_identity(root.parent, expected_tier="E1")


def _make_promotion_fixture(tmp_path: Path, *, visual: bool) -> tuple[Path, Path]:
    experiment = tmp_path / ("visual" if visual else "lora_only")
    checkpoint = experiment / "checkpoint-10"
    (experiment / "processor").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (experiment / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (experiment / "strategy_manifest.json").write_text(
        json.dumps({"strategy": "lora"}), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    if visual:
        (checkpoint / "visual_trainable_weights.safetensors").write_bytes(b"visual")
    return experiment, checkpoint


def test_r0_promotion_allows_lora_only_checkpoint_without_visual_sidecar(
    tmp_path: Path,
) -> None:
    module = _runner_module()
    experiment, _ = _make_promotion_fixture(tmp_path, visual=False)

    promoted = module._promote_half_checkpoint(
        experiment,
        10,
        require_visual_sidecar=False,
    )

    assert promoted.is_dir()
    assert (promoted / "adapter_model.safetensors").is_file()
    assert not (promoted / "visual_trainable_weights.safetensors").exists()


def test_r1_promotion_rejects_checkpoint_without_visual_sidecar(
    tmp_path: Path,
) -> None:
    module = _runner_module()
    experiment, _ = _make_promotion_fixture(tmp_path, visual=False)

    with pytest.raises(FileNotFoundError, match="visual sidecar is missing"):
        module._promote_half_checkpoint(experiment, 10, require_visual_sidecar=True)


def test_r1_promotion_accepts_checkpoint_with_visual_sidecar(tmp_path: Path) -> None:
    module = _runner_module()
    experiment, _ = _make_promotion_fixture(tmp_path, visual=True)

    promoted = module._promote_half_checkpoint(experiment, 10, require_visual_sidecar=True)

    assert (promoted / "visual_trainable_weights.safetensors").read_bytes() == b"visual"


def test_stage2_rejects_population_manifest_sha_mismatch(tmp_path: Path) -> None:
    module = _runner_module()
    population_manifest = tmp_path / "population_manifest.json"
    population_manifest.write_text("population-v1", encoding="utf-8")
    train_file = tmp_path / "stage2_train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    stage2_manifest = tmp_path / "stage2_manifest.json"
    stage2_manifest.write_text(
        json.dumps(
            {
                "sha256": module.sha256_file(train_file),
                "population_manifest_sha256": "stale-population-sha",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="population manifest SHA"):
        module._ensure_stage2(
            population_manifest,
            {
                "stage2": {
                    "output_file": str(train_file),
                    "manifest_file": str(stage2_manifest),
                }
            },
        )


def test_stage2_accepts_matching_population_manifest_sha(tmp_path: Path) -> None:
    module = _runner_module()
    population_manifest = tmp_path / "population_manifest.json"
    population_manifest.write_text("population-v1", encoding="utf-8")
    train_file = tmp_path / "stage2_train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    stage2_manifest = tmp_path / "stage2_manifest.json"
    manifest = {
        "sha256": module.sha256_file(train_file),
        "population_manifest_sha256": module.sha256_file(population_manifest),
    }
    stage2_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    _, returned_manifest, returned = module._ensure_stage2(
        population_manifest,
        {
            "stage2": {
                "output_file": str(train_file),
                "manifest_file": str(stage2_manifest),
            }
        },
    )

    assert returned_manifest == stage2_manifest
    assert returned["population_manifest_sha256"] == module.sha256_file(population_manifest)
