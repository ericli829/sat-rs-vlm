"""校验并安全解压 tar.gz 或 tar.zst 数据集归档。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算归档校验值。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """解析解压参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _verify_checksum(archive: Path, checksum_path: Path) -> None:
    expected = checksum_path.read_text(encoding="ascii").split()[0].lower()
    actual = file_sha256(archive)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {archive}: expected {expected}, got {actual}")


def _safe_extract(tar_path: Path, destination: Path, overwrite: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member escapes destination: {member.name}")
            if target.exists() and not overwrite:
                raise FileExistsError(f"Archive target already exists: {target}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in dataset archives: {member.name}")
        archive.extractall(root, filter="data")
        for member in archive.getmembers():
            if member.isfile():
                extracted = root / member.name
                if not extracted.is_file() or extracted.stat().st_size != member.size:
                    raise ValueError(
                        f"Post-extraction size check failed for archive member: {member.name}"
                    )


def main() -> int:
    """校验 checksum 后安全解压，失败时不静默继续。"""

    args = parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"Archive does not exist: {archive}")
    checksum = args.checksum or Path(f"{archive}.sha256")
    if checksum.is_file():
        _verify_checksum(archive, checksum)
    else:
        raise SystemExit(f"Checksum file does not exist: {checksum}")

    if str(archive).endswith(".tar.zst"):
        zstd = shutil.which("zstd")
        if zstd is None:
            raise SystemExit("tar.zst extraction requires the 'zstd' executable.")
        with tempfile.TemporaryDirectory(prefix="sat-rs-vlm-") as temp_dir:
            tar_path = Path(temp_dir) / "dataset.tar"
            completed = subprocess.run(
                [zstd, "-d", "-f", str(archive), "-o", str(tar_path)],
                check=False,
            )
            if completed.returncode != 0:
                raise SystemExit(f"zstd failed with exit code {completed.returncode}.")
            _safe_extract(tar_path, args.destination.resolve(), args.overwrite)
    else:
        _safe_extract(archive, args.destination.resolve(), args.overwrite)
    print(f"Extracted dataset to {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
