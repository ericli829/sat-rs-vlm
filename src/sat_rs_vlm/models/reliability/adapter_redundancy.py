"""Directory-level trusted recovery for deployable LoRA adapters.

A LoRA adapter is a versioned bundle, not merely adapter_model.safetensors.  This
module verifies every expected file against a trusted manifest and repairs only
from a warm/golden replica that independently matches that same manifest.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from sat_rs_vlm.models.reliability.checksum import (
    ChecksumManifest,
    build_checksum_manifest,
    file_sha256,
    verify_checksum_manifest,
    write_checksum_manifest,
)
from sat_rs_vlm.models.reliability.recovery import recover_file_from_backup


class AdapterScrubResult(BaseModel):
    success: bool
    working_root: str
    warm_root: str
    golden_root: str
    expected_files: int
    working_valid_before: bool
    warm_valid: bool
    golden_valid: bool
    restored_from_warm: list[str] = Field(default_factory=list)
    restored_from_golden: list[str] = Field(default_factory=list)
    unresolved_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / Path(*Path(relative).parts)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"manifest path escapes adapter root: {relative}")
    return candidate


def initialize_adapter_replicas(
    working_root: str | Path,
    *,
    warm_root: str | Path,
    golden_root: str | Path,
    manifest_path: str | Path,
) -> ChecksumManifest:
    """Create a trusted manifest for an already-created adapter replica chain."""

    working, warm, golden = _resolved(working_root), _resolved(warm_root), _resolved(golden_root)
    for path in (working, warm, golden):
        if not path.is_dir():
            raise NotADirectoryError(f"adapter replica directory missing: {path}")
    manifest = write_checksum_manifest(working, manifest_path)
    for replica in (warm, golden):
        verification = verify_checksum_manifest(manifest, root=replica)
        if not verification.valid:
            raise ValueError(f"replica does not match initialized working adapter: {replica}")
    return manifest


def scrub_adapter_replicas(
    working_root: str | Path,
    *,
    warm_root: str | Path,
    golden_root: str | Path,
    manifest: str | Path | ChecksumManifest,
) -> AdapterScrubResult:
    """Repair expected working adapter files from independently verified replicas.

    Warm has priority; golden is tried only when warm cannot be trusted.  No source
    copy is used unless the entire source directory matches the trusted manifest.
    Extra files are intentionally not deleted: deletion needs a deployment-specific
    retention policy and must not happen silently in a recovery routine.
    """

    working, warm, golden = _resolved(working_root), _resolved(warm_root), _resolved(golden_root)
    expected = manifest if isinstance(manifest, ChecksumManifest) else None
    working_check = verify_checksum_manifest(manifest, root=working)
    warm_check = verify_checksum_manifest(manifest, root=warm)
    golden_check = verify_checksum_manifest(manifest, root=golden)
    parsed = expected
    if parsed is None:
        from sat_rs_vlm.models.reliability.checksum import load_checksum_manifest

        parsed = load_checksum_manifest(manifest)

    result = AdapterScrubResult(
        success=False,
        working_root=str(working),
        warm_root=str(warm),
        golden_root=str(golden),
        expected_files=len(parsed.files),
        working_valid_before=working_check.valid,
        warm_valid=warm_check.valid,
        golden_valid=golden_check.valid,
    )
    if working_check.valid:
        result.success = True
        return result

    source_root = warm if warm_check.valid else golden if golden_check.valid else None
    source_name = "warm" if source_root == warm else "golden"
    if source_root is None:
        result.errors.append("no_trusted_adapter_recovery_source")
        result.unresolved_files = sorted({issue.path for issue in working_check.issues})
        return result

    expected_by_path = {entry.path: entry for entry in parsed.files}
    invalid_paths = {issue.path for issue in working_check.issues}
    for relative in sorted(invalid_paths):
        entry = expected_by_path.get(relative)
        if entry is None:
            continue
        source = _safe_child(source_root, relative)
        target = _safe_child(working, relative)
        recovery = recover_file_from_backup(target, source, expected_sha256=entry.sha256)
        if recovery.success:
            (
                result.restored_from_warm if source_name == "warm" else result.restored_from_golden
            ).append(relative)
        else:
            result.unresolved_files.append(relative)
            result.errors.extend(recovery.errors)

    after = verify_checksum_manifest(parsed, root=working)
    if after.valid:
        result.success = True
    else:
        result.unresolved_files = sorted(
            set(result.unresolved_files).union(issue.path for issue in after.issues)
        )
        if not result.errors:
            result.errors.append("adapter_recovery_incomplete")
    return result
