import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_convert_qwen3vl_script_generates_train_jsonl() -> None:
    prepare = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_rs_instruction_data.py",
            "--config",
            "configs/data/remote_sensing_data.yaml",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert prepare.returncode == 0, prepare.stderr
    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_to_qwen3vl_format.py",
            "--config",
            "configs/data/remote_sensing_data.yaml",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (ROOT / "data/processed/qwen3vl_train.jsonl").exists()
