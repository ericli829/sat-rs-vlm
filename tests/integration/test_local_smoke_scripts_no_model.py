import subprocess
import sys
from pathlib import Path

from sat_rs_vlm.utils.jsonl import write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def test_train_dry_run_without_loading_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    image_root = tmp_path / "data"
    image_root.mkdir()
    (image_root / "image.png").write_bytes(b"exists")
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    rows = [
        {
            "id": "s1",
            "task_type": "captioning",
            "images": ["image.png"],
            "instruction": "请描述图像。",
            "answer": "有建筑。",
        }
    ]
    write_jsonl(train_file, rows)
    write_jsonl(val_file, rows)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_qwen3vl_lora.py",
            "--config",
            "configs/train/qwen3vl_local_smoke.yaml",
            "--model-dir",
            str(model_dir),
            "--train-file",
            str(train_file),
            "--val-file",
            str(val_file),
            "--image-root",
            str(image_root),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run passed" in result.stdout
