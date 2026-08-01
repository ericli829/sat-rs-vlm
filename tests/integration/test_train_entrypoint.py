import os
import subprocess
import sys
from pathlib import Path

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
