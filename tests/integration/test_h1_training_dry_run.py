from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "scripts/train_qwen3vl_lora.py"


def test_h1_dry_run_requires_no_model_load(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    adapter = tmp_path / "stage-b-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    sample = {
        "id": "h1-fixture",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "task_type": "vqa",
        "metadata": {"dataset": "VRSBench"},
    }
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    content = json.dumps(sample) + "\n"
    train_file.write_text(content, encoding="utf-8")
    val_file.write_text(content, encoding="utf-8")
    config = {
        "model": {"model_dir": str(model), "processor_dir": str(model)},
        "data": {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "image_root": str(tmp_path),
            "max_seq_length": 1024,
        },
        "training": {
            "output_dir": str(tmp_path / "output"),
            "method": "lora",
            "num_train_epochs": None,
            "max_steps": 1,
        },
        "lora": {
            "initial_adapter_dir": str(adapter),
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "vision_tuning": {"enabled": True, "unfreeze_last_n_blocks": 2},
        "hard_adaptation": {
            "enabled": True,
            "hard_ratio": 0.7,
            "replay_ratio": 0.3,
        },
        "evaluation": {"do_eval": True},
    }
    config_path = tmp_path / "h1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path), "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Dry run passed. No model was loaded." in completed.stdout
    assert "Vision tuning enabled: True" in completed.stdout
