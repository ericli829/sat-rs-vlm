import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_check_env_returns_zero_with_base_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_env.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Python executable:" in result.stdout


def test_bootstrap_help_runs_without_installing() -> None:
    script = ROOT / "scripts" / "bootstrap_env.py"
    assert script.exists()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--with-model" in result.stdout
    assert "--torch-index-url" in result.stdout


def test_check_env_help_exposes_model_runtime_check() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_env.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--require-model" in result.stdout
