from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/data/prepare_dataset.py"


def test_prepare_dataset_rejects_missing_source_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset),
            "--train-file",
            str(tmp_path / "missing-train.jsonl"),
            "--validation-file",
            str(tmp_path / "missing-validation.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Source JSONL does not exist" in completed.stderr


def test_prepare_dataset_rejects_empty_required_source(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("", encoding="utf-8")
    validation.write_text("", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset),
            "--train-file",
            str(train),
            "--validation-file",
            str(validation),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Required source JSONL is empty" in completed.stderr
