from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.evaluate_rs_vlm import (
    FORMAL_GENERATION_MANIFEST_FIELDS,
    build_model_run_manifest,
    validate_formal_generation_config,
)


def _formal_config() -> dict[str, object]:
    return {
        "model": {
            "base_model": "Qwen3-VL-4B-Instruct",
            "adapter_path": "/models/adapter",
            "processor_id": "Qwen3-VL-4B-Instruct",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            "max_new_tokens": 256,
        },
        "prompt_reproducibility": {
            "require_complete": True,
            "generation_profile_name": "levir_caption_t1_t2_v1",
            "prompt_text": "Compare image T1 and image T2 and describe enduring changes.",
            "image_t1_role": "before",
            "image_t2_role": "after",
            "image_input_order": "T1_before_then_T2_after",
            "output_postprocessing": "decoded_text_strip_only",
            "model_id": "Qwen3-VL-4B-Instruct",
            "adapter_id": "adapter-20260823",
            "quantization": "none",
            "code_version": "feature/evaluation-v1.8-server-local-split@eb56b61",
        },
    }


def test_formal_generation_manifest_is_complete_and_hash_bound() -> None:
    config = _formal_config()
    project_root = Path(__file__).resolve().parents[3]
    predictions = project_root / "tests/fixtures/evaluation/visual_semantic_predictions.jsonl"
    config_path = project_root / "configs/eval/qwen3vl_eval.yaml"

    validate_formal_generation_config(config, checkpoint=None)
    manifest = build_model_run_manifest(
        config_path=config_path,
        eval_file=predictions,
        output_dir=project_root,
        config=config,
        checkpoint=None,
        predictions_file=predictions,
        summary_file=predictions,
        performance_file=predictions,
    )

    assert manifest["schema_version"] == "generation_manifest_v1"
    assert manifest["reproducibility_status"] == "complete"
    assert manifest["missing_reproducibility_fields"] == []
    assert manifest["temperature"] == "not_used_greedy"
    assert manifest["top_p"] == "not_used_greedy"
    assert all(manifest[field] is not None for field in FORMAL_GENERATION_MANIFEST_FIELDS)
    assert manifest["outputs"]["predictions_sha256"]
    json.dumps(manifest, ensure_ascii=False)


def test_formal_config_rejects_undeclared_prompt() -> None:
    config = _formal_config()
    reproducibility = dict(config["prompt_reproducibility"])
    reproducibility["prompt_text"] = None
    config["prompt_reproducibility"] = reproducibility

    with pytest.raises(ValueError, match="prompt_text_verbatim"):
        validate_formal_generation_config(config, checkpoint=None)
