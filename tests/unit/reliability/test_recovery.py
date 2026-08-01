from pathlib import Path

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.recovery import recover_file_from_backup


def test_checksum_backup_recovery_uses_atomic_replace(tmp_path: Path) -> None:
    backup = tmp_path / "backup.bin"
    target = tmp_path / "deployed.bin"
    backup.write_bytes(b"clean")
    target.write_bytes(b"fault")

    result = recover_file_from_backup(
        target,
        backup,
        expected_sha256=file_sha256(backup),
    )

    assert result.success
    assert result.used_atomic_replace
    assert target.read_bytes() == b"clean"
    assert backup.read_bytes() == b"clean"


def test_bad_backup_checksum_does_not_replace_target(tmp_path: Path) -> None:
    backup = tmp_path / "backup.bin"
    target = tmp_path / "deployed.bin"
    backup.write_bytes(b"wrong")
    target.write_bytes(b"fault")

    result = recover_file_from_backup(target, backup, expected_sha256="0" * 64)

    assert not result.success
    assert result.errors == ["backup_checksum_mismatch"]
    assert target.read_bytes() == b"fault"
