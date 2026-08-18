from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.sh"


def test_sweep_launcher_uses_autodl_python_and_stage_a_shutdown_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "activate_autodl_python" in text
    assert (
        '"$AUTODL_PYTHON" scripts/training/run_autodl_qwen3vl_4b_lr_merger_sweep.py'
        in text
    )
    assert "--shutdown-after-run" in text
    assert "/usr/sbin/shutdown -h now" in text
    assert "/sbin/shutdown -h now" in text
