import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/environment/bootstrap_local.py",
        "scripts/environment/check_environment.py",
        "scripts/environment/export_environment.py",
        "scripts/data/validate_dataset.py",
        "scripts/data/package_dataset.py",
        "scripts/data/unpack_dataset.py",
        "scripts/training/run_train.py",
        "scripts/training/run_smoke_train.py",
        "scripts/training/resume_train.py",
        "scripts/data/prepare_multisource_training_data.py",
    ],
)
def test_python_entrypoint_help(relative: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_shell_scripts_exist_and_parse_when_bash_is_available() -> None:
    scripts = [
        PROJECT_ROOT / "scripts/environment/setup_autodl.sh",
        PROJECT_ROOT / "scripts/environment/activate_autodl_python.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_train.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_full_pipeline.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_levircc_train.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_levircc_replay.sh",
        PROJECT_ROOT / "scripts/training/run_autodl_qwen3vl_4b_stage_a.sh",
        PROJECT_ROOT / "scripts/evaluation/run_autodl_replay_eval.sh",
        PROJECT_ROOT / "scripts/storage/sync_to_local_disk.sh",
        PROJECT_ROOT / "scripts/storage/backup_results.sh",
    ]
    assert all(path.is_file() for path in scripts)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed on this Windows host")
    for script in scripts:
        completed = subprocess.run(
            [bash, "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"
