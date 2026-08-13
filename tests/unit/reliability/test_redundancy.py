from pathlib import Path

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.redundancy import scrub_and_recover


def test_warm_replica_restores_corrupted_working_copy(tmp_path: Path) -> None:
    working, warm, golden = (tmp_path / "working.bin", tmp_path / "warm.bin", tmp_path / "golden.bin")
    for path in (working, warm, golden):
        path.write_bytes(b"trusted-model-weights")
    expected = file_sha256(working)
    working.write_bytes(b"corrupted")

    result = scrub_and_recover(
        working, warm_path=warm, golden_path=golden, expected_sha256=expected
    )

    assert result.success
    assert result.action == "restored"
    assert result.selected_source == "warm"
    assert working.read_bytes() == b"trusted-model-weights"


def test_golden_replica_is_used_when_warm_is_corrupted(tmp_path: Path) -> None:
    working, warm, golden = (tmp_path / "working.bin", tmp_path / "warm.bin", tmp_path / "golden.bin")
    for path in (working, warm, golden):
        path.write_bytes(b"trusted-model-weights")
    expected = file_sha256(working)
    working.write_bytes(b"bad-working")
    warm.write_bytes(b"bad-warm")

    result = scrub_and_recover(
        working, warm_path=warm, golden_path=golden, expected_sha256=expected
    )

    assert result.success
    assert result.selected_source == "golden"
    assert result.selected_tier == "golden"


def test_recovery_refuses_when_all_replicas_are_untrusted(tmp_path: Path) -> None:
    working, warm, golden = (tmp_path / "working.bin", tmp_path / "warm.bin", tmp_path / "golden.bin")
    for path in (working, warm, golden):
        path.write_bytes(b"bad")
    expected = "0" * 64

    result = scrub_and_recover(
        working, warm_path=warm, golden_path=golden, expected_sha256=expected
    )

    assert not result.success
    assert result.action == "failed"
    assert result.errors == ["no_trusted_recovery_source"]
