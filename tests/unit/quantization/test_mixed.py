from __future__ import annotations

import pytest

from sat_rs_vlm.quantization.mixed import build_mixed_precision_config


def test_mixed_config_preserves_sensitive_and_tied_modules() -> None:
    base_config = {
        "model": {"base_model": "model"},
        "quantization": {"backend": "bnb_int8", "device": "cuda"},
    }
    sensitivity_report = {
        "sensitive_groups": ["blocks_001"],
        "results": [
            {"name": "blocks_001", "module_names": ["model.layers.0.q_proj"]},
            {"name": "blocks_002", "module_names": ["model.layers.1.q_proj"]},
        ],
        "grouping": {"automatically_skipped_tied_linear_modules": ["lm_head"]},
    }

    mixed_config, summary = build_mixed_precision_config(base_config, sensitivity_report)

    assert mixed_config["quantization"]["llm_int8_skip_modules"] == [
        "lm_head",
        "model.layers.0.q_proj",
    ]
    assert summary["preserved_module_count"] == 2


def test_mixed_config_rejects_missing_sensitive_group() -> None:
    with pytest.raises(ValueError, match="missing from results"):
        build_mixed_precision_config(
            {"quantization": {}},
            {"sensitive_groups": ["missing"], "results": []},
        )


def test_mixed_config_can_preserve_top_group_without_threshold_match() -> None:
    mixed_config, summary = build_mixed_precision_config(
        {"quantization": {}},
        {
            "sensitive_groups": [],
            "results": [
                {"name": "lower", "sensitivity_score": 0.01, "module_names": ["lower.q"]},
                {"name": "higher", "sensitivity_score": 0.02, "module_names": ["higher.q"]},
            ],
        },
        keep_top_groups=1,
    )

    assert mixed_config["quantization"]["llm_int8_skip_modules"] == ["higher.q"]
    assert summary["sensitive_groups"] == ["higher"]
