from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_fake_external_plugin_dry_run_writes_inside_plugin(
    fake_plugin_root: Path, tmp_path: Path
) -> None:
    plugin = fake_plugin_root / "plugins" / "fake_strategy"
    model = tmp_path / "model"
    images = tmp_path / "images"
    model.mkdir()
    images.mkdir()
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    config = {
        "experiment": {"name": "fake-dry-run"},
        "model": {"model_dir": str(model), "processor_dir": str(model)},
        "data": {
            "train_file": str(train),
            "val_file": str(val),
            "image_root": str(images),
        },
        "training": {"max_steps": 1},
    }
    config_path = plugin / "configs" / "dry.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_external_strategy.py"),
            "--plugin-root",
            str(fake_plugin_root),
            "--strategy",
            "fake_strategy",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = plugin / "checkpoints" / "fake-dry-run"
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "train_report.json").is_file()
    assert not (PROJECT_ROOT / "checkpoints" / "fake-dry-run").exists()
