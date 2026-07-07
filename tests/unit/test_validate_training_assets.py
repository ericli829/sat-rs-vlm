from pathlib import Path

from scripts.validate_training_assets import validate_training_assets

from sat_rs_vlm.training.config import TrainingPathOverrides
from sat_rs_vlm.utils.jsonl import write_jsonl


def create_fake_assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    image_root = tmp_path / "data"
    image_root.mkdir()
    (image_root / "image.png").write_bytes(b"not-a-real-image-but-exists")
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
    return model_dir, train_file, val_file, image_root


def test_validate_assets_success_without_dependency_check(tmp_path: Path) -> None:
    model_dir, train_file, val_file, image_root = create_fake_assets(tmp_path)
    report_file = tmp_path / "report.json"
    report = validate_training_assets(
        "configs/train/qwen3vl_local_smoke.yaml",
        TrainingPathOverrides(
            model_dir=str(model_dir),
            train_file=str(train_file),
            val_file=str(val_file),
            image_root=str(image_root),
        ),
        check_model_dependencies=False,
        report_file=report_file,
    )
    assert report["success"] is True
    assert report_file.exists()


def test_validate_assets_reports_missing_path(tmp_path: Path) -> None:
    report = validate_training_assets(
        "configs/train/qwen3vl_local_smoke.yaml",
        TrainingPathOverrides(
            model_dir=str(tmp_path / "missing_model"),
            train_file=str(tmp_path / "missing_train.jsonl"),
            val_file=str(tmp_path / "missing_val.jsonl"),
            image_root=str(tmp_path),
        ),
        check_model_dependencies=False,
        report_file=tmp_path / "report.json",
    )
    assert report["success"] is False
    assert any("does not exist" in error for error in report["errors"])
