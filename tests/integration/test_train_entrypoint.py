import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "scripts/training/run_train.py"
CONFIG = PROJECT_ROOT / "configs/local/train_lora_smoke.yaml"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PROJECT_ROOT"] = str(PROJECT_ROOT)
    environment["DATA_ROOT"] = str(PROJECT_ROOT / "tests/fixtures/miniature_dataset")
    environment["MODEL_ROOT"] = str(PROJECT_ROOT / ".models")
    return environment


def test_mock_train_creates_reproducible_outputs(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--mock",
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / "config_resolved.yaml").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "logs/train.log").is_file()
    assert (output / "checkpoints/checkpoint-1/trainer_state.json").is_file()
    assert "/root/autodl" not in (output / "config_resolved.yaml").read_text(encoding="utf-8")


def test_mock_train_can_resume_latest(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    base = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(CONFIG),
        "--mock",
        "--output-dir",
        str(output),
    ]
    first = subprocess.run(base, cwd=PROJECT_ROOT, env=_environment(), check=False)
    second = subprocess.run(
        [*base, "--resume-latest"],
        cwd=PROJECT_ROOT,
        env=_environment(),
        check=False,
    )
    assert first.returncode == second.returncode == 0


def test_autodl_mock_forwards_structured_sampling_without_cuda(tmp_path: Path) -> None:
    config = tmp_path / "cloud-audit.yaml"
    config.write_text(
        """
experiment: {name: cloud_audit, seed: 7}
data:
  manifest_path: tests/fixtures/miniature_dataset/dataset_manifest.json
  max_train_samples: 3
  max_validation_samples: 1
  data_composition: detection_quota
  sampling_mode: weighted
  task_sampling_weights: {detection: 3.0, counting: 2.0, vqa: 1.0}
training: {max_steps: 1, gradient_accumulation_steps: 1, bf16: null, fp16: null}
runtime: {mock: true}
""",
        encoding="utf-8",
    )
    output = tmp_path / "cloud-experiment"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--environment",
            "autodl",
            "--env-config",
            str(PROJECT_ROOT / "configs/cloud/autodl.yaml"),
            "--mock",
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    legacy = yaml.safe_load(
        (output / "artifacts/legacy_training_config.yaml").read_text(encoding="utf-8")
    )
    assert legacy["data"]["data_composition"] == "detection_quota"
    assert legacy["data"]["sampling_mode"] == "weighted"
    assert legacy["data"]["task_sampling_weights"]["detection"] == 3.0
    assert legacy["training"]["dataloader_num_workers"] == 8
    assert legacy["training"]["dataloader_pin_memory"] is True
    assert legacy["training"]["dataloader_persistent_workers"] is True
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["precision"]["mode"] == "fp32"
