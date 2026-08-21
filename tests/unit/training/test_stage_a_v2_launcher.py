from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "training" / "run_autodl_qwen3vl_4b_stage_a_v2.sh"


def test_stage_a_v2_launcher_shutdown_is_explicit_and_mockable() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SHUTDOWN_AFTER_RUN=0" in text
    assert "--shutdown|--shutdown-after-run" in text
    assert "STAGE_A_V2_SHUTDOWN_MOCK_FILE" in text
    assert "--test-shutdown" in text
    assert 'os.system("/usr/bin/shutdown")' in text
    assert "/usr/sbin/shutdown -h now" in text
    assert "poweroff -f" in text


def test_stage_a_v2_launcher_never_shuts_down_diagnostics() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--prepare-only|--dry-run|--forward-only" in text
    assert "SHUTDOWN_AFTER_RUN=0" in text[text.index("[SAFE] Shutdown is disabled") :]


def test_stage_a_v2_launcher_exports_required_4b_runtime_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--model-dir" in text
    assert "--data-root" in text
    assert "--output-root" in text
    assert "Qwen3-VL-4B-Instruct" in text
    assert "export MODEL_ROOT DATA_ROOT OUTPUT_ROOT QWEN3VL_4B_MODEL_DIR" in text
    assert "Missing Qwen3-VL-4B model directory" in text


def test_stage_a_v2_launcher_does_not_create_its_own_screen_session() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "screen -dmS" not in text
