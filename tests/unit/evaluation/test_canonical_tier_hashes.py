from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.evaluation.tiers import canonical_jsonl_sha256, validate_tier_asset


def test_canonical_hash_ignores_json_whitespace_and_key_order(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"id": "a", "value": 1}\n{"id":"b","value":2}\n', encoding="utf-8")
    second.write_text('{"value":1,"id":"a"}\n { "value": 2, "id": "b" }\n', encoding="utf-8")
    assert canonical_jsonl_sha256(first) == canonical_jsonl_sha256(second)


def test_canonical_hash_preserves_row_order(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    second.write_text('{"id":"b"}\n{"id":"a"}\n', encoding="utf-8")
    assert canonical_jsonl_sha256(first) != canonical_jsonl_sha256(second)


def test_new_manifest_accepts_raw_drift_and_reports_warning(tmp_path: Path) -> None:
    tier = tmp_path / "e2.jsonl"
    tier.write_text('{"id":"a","value":1}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "tiers": {
                    "E2": {
                        "raw_sha256": "old-serialization",
                        "canonical_jsonl_sha256": canonical_jsonl_sha256(tier),
                        "sample_count": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = validate_tier_asset(tier="E2", eval_file=tier, manifest_path=manifest)
    assert result["provenance_warnings"]
    assert result["canonical_jsonl_sha256"] == canonical_jsonl_sha256(tier)


def test_new_manifest_canonical_mismatch_is_hard_failure(tmp_path: Path) -> None:
    tier = tmp_path / "e2.jsonl"
    tier.write_text('{"id":"a"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "tiers": {"E2": {"canonical_jsonl_sha256": "wrong", "sample_count": 1}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical JSONL SHA256 mismatch"):
        validate_tier_asset(tier="E2", eval_file=tier, manifest_path=manifest)


def test_legacy_manifest_raw_mismatch_remains_hard_failure(tmp_path: Path) -> None:
    tier = tmp_path / "e2.jsonl"
    tier.write_text('{"id":"a"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"tiers": {"E2": {"final_tier_sha256": "wrong", "sample_count": 1}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_tier_asset(tier="E2", eval_file=tier, manifest_path=manifest)


def test_formal_r1_reference_is_not_accepted_as_generated_unified_tier(tmp_path: Path) -> None:
    tier = tmp_path / "e2.jsonl"
    tier.write_text('{"id":"formal"}\n', encoding="utf-8")
    from sat_rs_vlm.evaluation.tiers import file_sha256

    formal_sha = file_sha256(tier)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tier_version": "unified-v2",
                "tiers": {
                    "E2": {
                        "sha256": "generated-sha",
                        "formal_r1_sha256": formal_sha,
                        "sample_count": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="formal R1 reference tier was supplied"):
        validate_tier_asset(tier="E2", eval_file=tier, manifest_path=manifest)
