import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_prepare_data_script_generates_sample_data() -> None:
    result = subprocess.run(
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
    assert result.returncode == 0, result.stderr
    assert (ROOT / "data/processed/rs_train.jsonl").exists()
    assert (ROOT / "data/processed/rs_val.jsonl").exists()
    assert (ROOT / "data/processed/rs_test.jsonl").exists()
