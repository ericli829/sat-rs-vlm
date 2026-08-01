"""流式 SHA-256、manifest 构建和验证。

所有文件按固定大小分块读取，适用于大型模型权重。manifest 只保存相对路径、文件大小
和摘要，不包含本地或云端绝对路径。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path, PurePosixPath

from sat_rs_vlm.models.reliability.schemas import (
    ChecksumEntry,
    ChecksumIssue,
    ChecksumManifest,
    ChecksumVerificationResult,
)

CHUNK_SIZE = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """按 1 MiB 分块计算文件 SHA-256，并返回十六进制摘要。"""

    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_files(root: Path, excluded: set[Path]) -> Iterable[Path]:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() not in excluded:
            yield path


def build_checksum_manifest(
    root: str | Path,
    *,
    exclude: Iterable[str | Path] = (),
) -> ChecksumManifest:
    """扫描目录并在内存中构建 checksum manifest。

    参数：
        root：待保护目录。
        exclude：不纳入 manifest 的文件，例如 manifest 输出文件本身。

    返回值：
        `ChecksumManifest`，其中每个 `path` 都相对 `root` 且使用 POSIX 分隔符。
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Checksum root is not a directory: {root_path}")
    excluded = {Path(path).expanduser().resolve() for path in exclude}
    entries = [
        ChecksumEntry(
            path=path.relative_to(root_path).as_posix(),
            size=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in _relative_files(root_path, excluded)
    ]
    return ChecksumManifest(files=entries)


def write_checksum_manifest(root: str | Path, output: str | Path) -> ChecksumManifest:
    """构建并写出 manifest；`root` 字段相对 manifest 所在目录。"""

    root_path = Path(root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    manifest = build_checksum_manifest(root_path, exclude=(output_path,))
    manifest.root = Path(os.path.relpath(root_path, output_path.parent)).as_posix()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_checksum_manifest(path: str | Path) -> ChecksumManifest:
    """读取并校验 checksum manifest schema。"""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ChecksumManifest.model_validate(payload)


def _safe_manifest_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Manifest file path must stay relative to root: {relative}")
    candidate = (root / Path(*pure.parts)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Manifest file path escapes root: {relative}")
    return candidate


def verify_checksum_manifest(
    manifest: str | Path | ChecksumManifest,
    *,
    root: str | Path | None = None,
) -> ChecksumVerificationResult:
    """验证文件缺失、大小变化和 SHA-256 不匹配并返回全部问题。"""

    if isinstance(manifest, ChecksumManifest):
        parsed = manifest
        if root is None:
            raise ValueError("root is required when verifying an in-memory manifest")
        root_path = Path(root).expanduser().resolve()
    else:
        manifest_path = Path(manifest).expanduser().resolve()
        parsed = load_checksum_manifest(manifest_path)
        root_path = (
            Path(root).expanduser().resolve()
            if root is not None
            else (manifest_path.parent / Path(parsed.root)).resolve()
        )
    if not root_path.is_dir():
        raise NotADirectoryError(f"Checksum root is not a directory: {root_path}")

    issues: list[ChecksumIssue] = []
    for entry in parsed.files:
        path = _safe_manifest_path(root_path, entry.path)
        if not path.is_file():
            issues.append(ChecksumIssue(path=entry.path, code="missing", expected=entry.sha256))
            continue
        actual_size = path.stat().st_size
        if actual_size != entry.size:
            issues.append(
                ChecksumIssue(
                    path=entry.path,
                    code="size_mismatch",
                    expected=entry.size,
                    actual=actual_size,
                )
            )
        actual_hash = file_sha256(path)
        if actual_hash != entry.sha256:
            issues.append(
                ChecksumIssue(
                    path=entry.path,
                    code="hash_mismatch",
                    expected=entry.sha256,
                    actual=actual_hash,
                )
            )
    return ChecksumVerificationResult(
        valid=not issues,
        root=str(root_path),
        checked_files=len(parsed.files),
        issues=issues,
    )
