"""把数据集打包为 tar.gz 或可选 tar.zst，并生成 SHA-256。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256，避免大数据包全部载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """解析打包参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("tar.gz", "tar.zst"), default="tar.gz")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def package_dataset(source: Path, output: Path, archive_format: str) -> None:
    """按指定格式创建保持数据集顶层目录的归档。"""

    if archive_format == "tar.gz":
        with tarfile.open(output, "w:gz") as archive:
            archive.add(source, arcname=source.name)
        return
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("tar.zst requires the 'zstd' executable; use --format tar.gz.")
    with tempfile.TemporaryDirectory(prefix="sat-rs-vlm-") as temp_dir:
        tar_path = Path(temp_dir) / f"{source.name}.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(source, arcname=source.name)
        completed = subprocess.run(
            [zstd, "-T0", "-f", str(tar_path), "-o", str(output)],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"zstd failed with exit code {completed.returncode}.")


def main() -> int:
    """执行打包并写入 `<archive>.sha256`。"""

    args = parse_args()
    source = args.dataset_root.resolve()
    if not source.is_dir():
        raise SystemExit(f"Dataset root does not exist: {source}")
    output = args.output.resolve()
    expected_suffix = ".tar.gz" if args.format == "tar.gz" else ".tar.zst"
    if not str(output).endswith(expected_suffix):
        raise SystemExit(f"Output must end with {expected_suffix}: {output}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Archive already exists: {output}. Pass --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    package_dataset(source, output, args.format)
    checksum = file_sha256(output)
    checksum_path = Path(f"{output}.sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    print(f"Archive: {output}")
    print(f"SHA-256: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
