from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgraph_lab.tools.generate_teacher_batches import _validate_or_write_manifest


def manifest() -> dict:
    return {
        "version": "taskgraph-stage1-batch-generation-v1",
        "created_at": "fixture",
        "schema_version": "taskgraph-v1.1",
        "prompt_version": "fixture-prompt",
        "input_sha256": "input",
        "config_sha256": "config",
        "system_prompt_sha256": "prompt",
        "batch_contract_sha256": "batch",
        "batch_size": 4,
        "provider": "fixture",
        "model": "fixture",
        "thinking": "enabled",
        "reasoning_effort": "low",
    }


def test_stage1_manifest_supports_matching_resume(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    current = manifest()
    _validate_or_write_manifest(path, current)
    _validate_or_write_manifest(path, current | {"created_at": "later"})
    assert json.loads(path.read_text(encoding="utf-8"))["created_at"] == "fixture"


def test_stage1_manifest_rejects_different_prompt_or_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    _validate_or_write_manifest(path, manifest())
    with pytest.raises(ValueError, match="provenance"):
        _validate_or_write_manifest(path, manifest() | {"batch_size": 8})
    with pytest.raises(ValueError, match="provenance"):
        _validate_or_write_manifest(path, manifest() | {"system_prompt_sha256": "different"})
