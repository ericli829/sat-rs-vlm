from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[3]


def _call_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def test_lora_training_entry_uses_canonical_multitask_trainer_and_metadata() -> None:
    source = (PROJECT_ROOT / "scripts/train_qwen3vl_lora.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    calls = [node for node in ast.walk(train_function) if isinstance(node, ast.Call)]
    trainer_calls = [call for call in calls if _call_name(call) == "create_multitask_trainer"]

    assert len(trainer_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in trainer_calls[0].keywords}
    assert ast.unparse(keywords["loss_config"]) == "config.loss"
    assert "checkpoint_artifact_saver" in keywords
    assert all(_call_name(call) != "create_trainer" for call in calls)
    collator_calls = [call for call in calls if _call_name(call) == "Qwen3VLDataCollator"]
    assert any(
        isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for call in collator_calls
        for keyword in call.keywords
        if keyword.arg == "include_task_metadata"
    )


def test_canonical_trainer_compute_loss_calls_configured_strategy() -> None:
    source = (PROJECT_ROOT / "src/sat_rs_vlm/training/trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    compute_loss_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and _call_name(call) == "compute_multitask_loss"
    ]

    assert len(compute_loss_calls) == 1
    assert ast.unparse(compute_loss_calls[0].args[3]).endswith("loss_config")
