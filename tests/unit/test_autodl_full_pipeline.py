from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/training/run_autodl_full_pipeline.sh"


def test_full_pipeline_orders_evaluation_before_backup_and_shutdown() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    evaluation = text.index("python scripts/evaluate_rs_vlm.py")
    backup = text.rindex("bash scripts/storage/backup_results.sh")
    shutdown = text.rindex("sudo shutdown -h now")
    assert evaluation < backup < shutdown
    assert "trap on_error ERR" in text
    assert "full validation-set generation evaluation" in text


def test_full_pipeline_saves_error_report_before_post_training_shutdown() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    report = text.index("full_pipeline_error_")
    shutdown = text.index("sudo shutdown -h now || true")
    assert report < shutdown
    assert "TRAINING_COMPLETED=1" in text
    assert "Training did not complete; the AutoDL instance will remain running." in text


def test_full_pipeline_uses_fixed_autodl_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'PROJECT_ROOT="/root/autodl-tmp/sat-rs-vlm"' in text
    assert 'DATA_ROOT="/root/autodl-tmp/datasets"' in text
    assert 'MODEL_ROOT="/root/autodl-tmp/models"' in text
    assert 'OUTPUT_ROOT="/root/autodl-tmp/outputs"' in text
    assert 'BACKUP_ROOT="/root/autodl-fs/experiments"' in text
