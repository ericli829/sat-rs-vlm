from __future__ import annotations

from pathlib import Path

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts"
    / "training"
    / "run_autodl_qwen3vl_4b_stage_a.sh"
)


def test_stage_a_launcher_reports_failure_before_optional_shutdown() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "trap on_error ERR" in text
    report = text.index("stage_a_error_")
    sync = text.index("sync || true", report)
    shutdown = text.index("shutdown_host || true", sync)
    assert report < sync < shutdown
    assert "last_valid_adapter=" in text
    assert "resume_hint=" in text


def test_stage_a_launcher_shutdown_is_explicit_and_root_compatible() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--shutdown-after-run" in text
    assert "SHUTDOWN_AFTER_RUN=0" in text
    assert "/usr/sbin/shutdown -h now" in text
    assert "/sbin/shutdown -h now" in text
