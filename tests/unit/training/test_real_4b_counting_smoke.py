from scripts.training.run_real_4b_counting_smoke import (
    build_real4bit_fallback_config,
    is_local_low_memory_failure,
    validated_low_memory_reason,
)
from transformers.integrations.bitsandbytes import should_convert_module


def test_real_4b_smoke_nf4_fallback_is_limited_to_memory_failures() -> None:
    assert is_local_low_memory_failure("CUDA out of memory")
    assert is_local_low_memory_failure(
        "We need an `offload_dir` to dispatch this model according to this device_map"
    )
    assert not is_local_low_memory_failure("R1 provenance SHA mismatch")
    assert not is_local_low_memory_failure("Expert step-0 parity failed")


def test_nf4_visual_skip_preserves_all_formal_sidecar_linear_modules() -> None:
    skip = ["model.visual", "lm_head"]
    assert not should_convert_module("model.visual.blocks.22.attn.qkv", skip)
    assert not should_convert_module("model.visual.merger.linear_fc1", skip)
    assert not should_convert_module("lm_head", skip)
    assert should_convert_module("model.language_model.layers.0.self_attn.q_proj", skip)


def test_real4bit_fallback_keeps_source_config_immutable_and_r1_additive() -> None:
    source = {"model": {"r1_integration": "merge"}, "training": {"bf16": True}}
    fallback = build_real4bit_fallback_config(source)
    assert source["model"] == {"r1_integration": "merge"}
    assert fallback["model"] == {
        "r1_integration": "additive",
        "load_in_4bit": True,
        "quantization_skip_modules": ["model.visual", "lm_head"],
    }


def test_direct_nf4_requires_a_validated_bf16_low_memory_log(tmp_path) -> None:
    evidence = tmp_path / "bf16.log"
    evidence.write_text("We need an `offload_dir` to dispatch this model", encoding="utf-8")
    reason = validated_low_memory_reason(str(evidence))
    assert reason is not None and reason.endswith("need an `offload_dir` to dispatch this model")

    invalid = tmp_path / "invalid.log"
    invalid.write_text("provenance mismatch", encoding="utf-8")
    try:
        validated_low_memory_reason(str(invalid))
    except ValueError as exc:
        assert "no accepted marker" in str(exc)
    else:
        raise AssertionError("Invalid low-memory evidence was accepted")
