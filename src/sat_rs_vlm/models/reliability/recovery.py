"""checksum 检测后的文件级原子恢复。

恢复先校验备份，复制到部署文件同目录的临时文件，再校验临时文件，最后通过
`os.replace` 原子替换目标。任何替换前错误都不会破坏当前部署文件。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.protection import output_guard_vote
from sat_rs_vlm.models.reliability.schemas import RecoveryResult


def recover_file_from_backup(
    target_path: str | Path,
    backup_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> RecoveryResult:
    """从干净备份恢复单文件并返回恢复前后 hash 和稳定错误码。"""

    target = Path(target_path).expanduser().resolve()
    backup = Path(backup_path).expanduser().resolve()
    if target == backup:
        raise ValueError("target_path and backup_path must be different")
    before_hash = file_sha256(target) if target.is_file() else None
    if not backup.is_file():
        return RecoveryResult(
            success=False,
            target_path=str(target),
            backup_path=str(backup),
            expected_sha256=expected_sha256 or "",
            before_sha256=before_hash,
            errors=["backup_missing"],
        )
    backup_hash = file_sha256(backup)
    expected = expected_sha256 or backup_hash
    if backup_hash != expected:
        return RecoveryResult(
            success=False,
            target_path=str(target),
            backup_path=str(backup),
            expected_sha256=expected,
            before_sha256=before_hash,
            backup_sha256=backup_hash,
            errors=["backup_checksum_mismatch"],
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.recovery-",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(backup, temporary)
        if file_sha256(temporary) != expected:
            return RecoveryResult(
                success=False,
                target_path=str(target),
                backup_path=str(backup),
                expected_sha256=expected,
                before_sha256=before_hash,
                backup_sha256=backup_hash,
                errors=["temporary_checksum_mismatch"],
            )
        os.replace(temporary, target)
        after_hash = file_sha256(target)
        return RecoveryResult(
            success=after_hash == expected,
            target_path=str(target),
            backup_path=str(backup),
            expected_sha256=expected,
            before_sha256=before_hash,
            backup_sha256=backup_hash,
            after_sha256=after_hash,
            used_atomic_replace=True,
            errors=[] if after_hash == expected else ["recovery_checksum_mismatch"],
        )
    except OSError:
        return RecoveryResult(
            success=False,
            target_path=str(target),
            backup_path=str(backup),
            expected_sha256=expected,
            before_sha256=before_hash,
            backup_sha256=backup_hash,
            errors=["recovery_io_error"],
        )
    finally:
        temporary.unlink(missing_ok=True)


def guarded_select_prediction(
    task_type: str,
    predictions: list[str],
    *,
    fallback: str = "",
) -> dict[str, object]:
    """兼容旧调用：使用统一 output guard 后返回普通字典。"""

    return output_guard_vote(task_type, predictions, fallback=fallback).model_dump(mode="json")
