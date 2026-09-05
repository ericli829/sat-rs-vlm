from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT_ROOT / "data/evaluation/tiers_v2/e_count_v2_manifest.json"
TIER = PROJECT_ROOT / "data/evaluation/tiers_v2/e_count_v2.jsonl"


def test_formal_r1_327_exact_cardinality_ids_are_in_e_count_v2() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in TIER.open(encoding="utf-8") if line.strip()]
    row_ids = {str(row["id"]) for row in rows}
    valid_ids = list(manifest["exact_cardinality_valid_sample_ids"])

    assert manifest["tier_name"] == "E_COUNT_V2"
    assert manifest["tier_version"] == "unified-v2"
    assert manifest["sources"]["E2"]["formal_r1_sha256"] == (
        "230b23a655d8973ec3a165041ecc0d7e23043340add5b3eb4c1df0b65d4436cc"
    )
    assert manifest["raw_counting_count"] == 377
    assert manifest["exact_cardinality_valid_count"] == 327
    assert len(valid_ids) == len(set(valid_ids)) == 327
    assert manifest["exact_cardinality_valid_sample_ids_sha256"] == (
        "9194a0f54f1b2e7d71d7ea595ea22a8873374022de740eb5e42f5c45aaf7a60e"
    )
    assert set(valid_ids) <= set(manifest["counting_sample_ids"])
    assert set(valid_ids) <= row_ids
