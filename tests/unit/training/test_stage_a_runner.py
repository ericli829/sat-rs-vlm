from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/training/run_qwen3vl_4b_stage_a.py"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_qwen3vl_4b_stage_a", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round_command_chains_previous_adapter_without_changing_processor() -> None:
    module = load_runner()
    args = SimpleNamespace(
        train_config="stage-a.yaml",
        max_train_samples=None,
        max_eval_samples=None,
    )
    command = module._training_command(
        args,
        train_file=Path("round_001.jsonl"),
        validation_file=Path("validation.jsonl"),
        output_dir=Path("round_001/adapter"),
        initial_adapter=Path("round_000/adapter"),
        learning_rate=1.0e-5,
        mode=None,
    )
    adapter_index = command.index("--initial-adapter")
    assert command[adapter_index + 1] == str(Path("round_000/adapter"))
    assert "--save-steps" not in command
    assert "--processor-dir" not in command


def test_round_command_can_resolve_dynamic_mid_round_save_step() -> None:
    module = load_runner()
    args = SimpleNamespace(
        train_config="stage-a.yaml",
        max_train_samples=None,
        max_eval_samples=None,
    )
    command = module._training_command(
        args,
        train_file=Path("round_001.jsonl"),
        validation_file=Path("validation.jsonl"),
        output_dir=Path("round_001/adapter"),
        initial_adapter=Path("round_000/adapter"),
        learning_rate=1.0e-5,
        mode="--dry-run",
        save_steps=102,
    )
    save_index = command.index("--save-steps")
    assert command[save_index + 1] == "102"
    assert command[-1] == "--dry-run"


def test_adapter_chain_starts_from_base_then_uses_immediately_previous_round() -> None:
    module = load_runner()
    root = Path("run")
    assert module._previous_round_adapter(root, 0) is None
    assert module._previous_round_adapter(root, 1) == root / "round_000/adapter"
    assert module._previous_round_adapter(root, 2) == root / "round_001/adapter"


def test_round_learning_rate_reuses_last_configured_value(tmp_path: Path) -> None:
    module = load_runner()
    config = tmp_path / "config.yaml"
    config.write_text(
        "cycle_training:\n  learning_rates: [0.00002, 0.00001]\n", encoding="utf-8"
    )
    assert module._learning_rate(config, 0, None) == 2.0e-5
    assert module._learning_rate(config, 4, None) == 1.0e-5
    assert module._learning_rate(config, 4, 3.0e-6) == 3.0e-6


def test_round_plan_uses_world_size_and_half_epoch_checkpoint() -> None:
    module = load_runner()
    plan = module._resolve_round_plan(
        {"sample_count": 52211},
        {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4},
        world_size=1,
    )
    assert plan == {
        "sample_count": 52211,
        "world_size": 1,
        "effective_batch": 16,
        "steps_per_epoch": 3264,
        "mid_round_save_step": 1632,
        "final_step": 3264,
    }


def test_initial_adapter_validation_rejects_2b_fingerprint(tmp_path: Path) -> None:
    module = load_runner()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3_vl", "architectures":["Qwen3VLForConditionalGeneration"],'
        '"text_config":{"hidden_size":2560,"num_hidden_layers":36,"num_attention_heads":32,"vocab_size":151936},'
        '"vision_config":{"hidden_size":1024,"depth":24}}',
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "strategy_manifest.json").write_text(
        '{"strategy":"lora", "base_model_fingerprint":'
        '{"model_type":"qwen3_vl", "hidden_size":1536, "num_hidden_layers":28,'
        '"num_attention_heads":16, "vocab_size":151936, "vision_hidden_size":1024,'
        '"vision_depth":24}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="2B/4B mismatch"):
        module._validate_initial_adapter(adapter, model_dir, require_fingerprint=True)


def test_initial_adapter_validation_rejects_h1_h2_provenance(tmp_path: Path) -> None:
    module = load_runner()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"qwen3_vl", "text_config":{"hidden_size":2560,'
        '"num_hidden_layers":36,"num_attention_heads":32,"vocab_size":151936},'
        '"vision_config":{"hidden_size":1024,"depth":24}}',
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "strategy_manifest.json").write_text(
        '{"strategy":"lora", "training_stage":"h1_hard_example_visual_adaptation",'
        '"base_model_fingerprint":{"model_type":"qwen3_vl", "hidden_size":2560,'
        '"num_hidden_layers":36, "num_attention_heads":32, "vocab_size":151936,'
        '"vision_hidden_size":1024, "vision_depth":24}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="specialized training-stage"):
        module._validate_initial_adapter(adapter, model_dir, require_fingerprint=True)


def test_round_source_contract_accepts_replay_top_up(tmp_path: Path) -> None:
    module = load_runner()
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        "  source_batch_pattern: [VRSBench, VRSBench, VRSBench, LEVIR-CC]\n",
        encoding="utf-8",
    )
    rounds = [
        {"source_distribution": {"VRSBench": 12, "LEVIR-CC": 4}},
        {"source_distribution": {"VRSBench": 10, "LEVIR-CC": 4}},
    ]

    module._validate_round_source_contract(
        rounds,
        config,
        start_round=0,
        end_round=1,
    )


def test_round_source_contract_rejects_missing_short_source(tmp_path: Path) -> None:
    module = load_runner()
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        "  source_batch_pattern: [VRSBench, VRSBench, VRSBench, LEVIR-CC]\n",
        encoding="utf-8",
    )
    rounds = [{"source_distribution": {"VRSBench": 12}}]

    with pytest.raises(ValueError, match="missing.*LEVIR-CC"):
        module._validate_round_source_contract(
            rounds,
            config,
            start_round=0,
            end_round=0,
        )


def test_changed_cycle_manifest_is_archived_before_resume(tmp_path: Path) -> None:
    module = load_runner()
    run_root = tmp_path / "run"
    run_root.mkdir()
    destination = run_root / "cycle_manifest.json"
    destination.write_text('{"version":"old"}\n', encoding="utf-8")
    new_manifest = tmp_path / "new_manifest.json"
    new_manifest.write_text('{"version":"new"}\n', encoding="utf-8")

    archived = module._store_cycle_manifest(new_manifest, run_root)

    assert len(archived) == 1
    assert Path(archived[0]).read_text(encoding="utf-8") == '{"version":"old"}\n'
    assert destination.read_text(encoding="utf-8") == '{"version":"new"}\n'
