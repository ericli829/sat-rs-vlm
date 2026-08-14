from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from sat_rs_vlm.evaluation.tiers import (
    DEFAULT_EVALUATION_TIER,
    default_tier_file,
    normalize_tier,
    validate_tier_asset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_formal_evaluation_config_defaults_to_e2() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "configs/eval/qwen3vl_eval.yaml").read_text())
    assert payload["evaluation"]["tier"] == DEFAULT_EVALUATION_TIER == "E2"
    assert payload["data"]["eval_file"] == default_tier_file("E2")
    assert payload["data"]["max_eval_samples"] is None


def test_explicit_tier_configs_are_not_implicitly_downgraded() -> None:
    for tier in ("E1", "E2", "E3"):
        path = PROJECT_ROOT / f"configs/eval/qwen3vl_eval_{tier.lower()}.yaml"
        payload = yaml.safe_load(path.read_text())
        assert payload["evaluation"]["tier"] == tier
        assert payload["data"]["eval_file"] == default_tier_file(tier)
        assert payload["data"]["max_eval_samples"] is None


def test_tier_asset_hash_and_count_are_verified(tmp_path: Path) -> None:
    tier_file = tmp_path / "e2_standard.jsonl"
    tier_file.write_text('{"id":"sample-1"}\n{"id":"sample-2"}\n', encoding="utf-8")
    digest = hashlib.sha256(tier_file.read_bytes()).hexdigest()
    manifest = tmp_path / "evaluation_tiers_manifest.json"
    manifest.write_text(
        json.dumps({"tiers": {"E2": {"sha256": digest, "sample_count": 2}}}),
        encoding="utf-8",
    )
    result = validate_tier_asset(tier="E2", eval_file=tier_file, manifest_path=manifest)
    assert result["tier"] == "E2"
    assert result["sha256"] == digest
    assert normalize_tier(None) == "E2"
