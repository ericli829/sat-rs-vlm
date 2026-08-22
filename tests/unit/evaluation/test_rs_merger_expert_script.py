import yaml
from scripts.evaluation.evaluate_rs_merger_expert import (
    configure_decoder_only_generation_padding,
    resolve_controller_spec,
    resolve_real4bit_skip_modules,
)


def test_generation_evaluator_uses_left_padding() -> None:
    processor = type("Processor", (), {})()
    processor.tokenizer = type("Tokenizer", (), {"padding_side": "right"})()
    configure_decoder_only_generation_padding(processor)
    assert processor.tokenizer.padding_side == "left"


def test_c4_evaluation_restores_wide_detail_architecture() -> None:
    assert resolve_controller_spec(
        {
            "variant": "c4_wide",
            "detail_hidden_size": 1024,
            "interface_lora_parameter_count": 0,
        }
    ) == {
        "variant": "rs_detail",
        "detail_hidden_size": 1024,
        "interface_lora_enabled": False,
    }


def test_real4bit_evaluation_restores_checkpoint_quantization_exclusions(tmp_path) -> None:
    (tmp_path / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "load_in_4bit": True,
                    "quantization_skip_modules": ["model.visual", "lm_head"],
                }
            }
        ),
        encoding="utf-8",
    )
    assert resolve_real4bit_skip_modules(
        tmp_path, {"foundation_precision": "real_4bit_nf4_bf16_compute"}
    ) == ["model.visual", "lm_head"]
    assert resolve_real4bit_skip_modules(tmp_path, {"foundation_precision": "bf16"}) is None
