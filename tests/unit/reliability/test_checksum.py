from pathlib import Path

from sat_rs_vlm.models.reliability.checksum import (
    verify_checksum_manifest,
    write_checksum_manifest,
)


def test_checksum_manifest_build_and_verify(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "a.bin").write_bytes(b"abc")
    (root / "nested").mkdir()
    (root / "nested/b.json").write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "checksums.json"

    manifest = write_checksum_manifest(root, manifest_path)
    result = verify_checksum_manifest(manifest_path)

    assert [entry.path for entry in manifest.files] == ["a.bin", "nested/b.json"]
    assert result.valid
    assert result.checked_files == 2


def test_checksum_detects_size_hash_and_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    root.mkdir()
    first = root / "first.bin"
    second = root / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest_path = root / "checksums.json"
    write_checksum_manifest(root, manifest_path)

    first.write_bytes(b"changed-size")
    second.unlink()
    result = verify_checksum_manifest(manifest_path)
    codes = {(issue.path, issue.code) for issue in result.issues}

    assert not result.valid
    assert ("first.bin", "size_mismatch") in codes
    assert ("first.bin", "hash_mismatch") in codes
    assert ("second.bin", "missing") in codes
