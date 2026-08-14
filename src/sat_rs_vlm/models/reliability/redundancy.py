"""Multi-copy trusted recovery for model files.

This software component models a deployed working copy, a warm verification copy,
and a rarely touched golden copy.  It does not claim hardware radiation immunity:
every candidate must independently match a trusted SHA-256 before it may restore
another copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.recovery import recover_file_from_backup

ReplicaTier = Literal["working", "warm", "golden"]


class ReplicaStatus(BaseModel):
    name: str
    tier: ReplicaTier
    path: str
    exists: bool
    sha256: str | None = None
    trusted: bool = False
    error: str | None = None


class ReplicaRecoveryResult(BaseModel):
    success: bool
    expected_sha256: str
    target_before_sha256: str | None = None
    target_after_sha256: str | None = None
    selected_source: str | None = None
    selected_tier: ReplicaTier | None = None
    action: Literal["no_action", "restored", "failed"]
    replicas: list[ReplicaStatus] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def inspect_replicas(
    replicas: list[tuple[str, ReplicaTier, str | Path]],
    *,
    expected_sha256: str,
) -> list[ReplicaStatus]:
    """Verify every copy independently against a trusted expected digest."""

    statuses: list[ReplicaStatus] = []
    for name, tier, raw_path in replicas:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            statuses.append(
                ReplicaStatus(name=name, tier=tier, path=str(path), exists=False, error="missing")
            )
            continue
        digest = file_sha256(path)
        statuses.append(
            ReplicaStatus(
                name=name,
                tier=tier,
                path=str(path),
                exists=True,
                sha256=digest,
                trusted=digest == expected_sha256,
                error=None if digest == expected_sha256 else "checksum_mismatch",
            )
        )
    return statuses


def scrub_and_recover(
    working_path: str | Path,
    *,
    warm_path: str | Path,
    golden_path: str | Path,
    expected_sha256: str,
) -> ReplicaRecoveryResult:
    """Verify working/warm/golden copies and atomically restore only from a trusted copy.

    Priority is warm then golden.  If neither backup verifies, the function refuses
    recovery and leaves the working file unchanged.
    """

    working = Path(working_path).expanduser().resolve()
    statuses = inspect_replicas(
        [
            ("working", "working", working),
            ("warm", "warm", warm_path),
            ("golden", "golden", golden_path),
        ],
        expected_sha256=expected_sha256,
    )
    working_status, *backups = statuses
    if working_status.trusted:
        return ReplicaRecoveryResult(
            success=True,
            expected_sha256=expected_sha256,
            target_before_sha256=working_status.sha256,
            target_after_sha256=working_status.sha256,
            action="no_action",
            replicas=statuses,
        )
    source = next((status for status in backups if status.trusted), None)
    if source is None:
        return ReplicaRecoveryResult(
            success=False,
            expected_sha256=expected_sha256,
            target_before_sha256=working_status.sha256,
            action="failed",
            replicas=statuses,
            errors=["no_trusted_recovery_source"],
        )
    recovery = recover_file_from_backup(working, source.path, expected_sha256=expected_sha256)
    return ReplicaRecoveryResult(
        success=recovery.success,
        expected_sha256=expected_sha256,
        target_before_sha256=recovery.before_sha256,
        target_after_sha256=recovery.after_sha256,
        selected_source=source.name,
        selected_tier=source.tier,
        action="restored" if recovery.success else "failed",
        replicas=statuses,
        errors=recovery.errors,
    )
