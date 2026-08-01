from pathlib import Path
from types import SimpleNamespace

from scripts.train_qwen3vl_lora import build_strategy_manifest

from sat_rs_vlm.training.config import ResolvedTrainingPaths


class _Parameter:
    dtype = "torch.bfloat16"


class _Model:
    def parameters(self):  # type: ignore[no-untyped-def]
        return iter([_Parameter()])

    def named_modules(self):  # type: ignore[no-untyped-def]
        return iter(
            [
                ("base_model.model.layers.0.self_attn.q_proj", object()),
                ("base_model.model.layers.0.self_attn.v_proj", object()),
            ]
        )


def test_lora_manifest_is_compatible_with_checkpoint_loader_contract(tmp_path: Path) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(method="lora"),
        lora=SimpleNamespace(target_modules=["q_proj", "v_proj"]),
        qlora=SimpleNamespace(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        ),
    )
    paths = ResolvedTrainingPaths(
        model_source="/models/Qwen3-VL-2B-Instruct",
        processor_source="/models/Qwen3-VL-2B-Instruct",
        model_dir=Path("/models/Qwen3-VL-2B-Instruct"),
        processor_dir=Path("/models/Qwen3-VL-2B-Instruct"),
        train_file=tmp_path / "train.jsonl",
        val_file=tmp_path / "val.jsonl",
        image_root=tmp_path,
        output_dir=tmp_path / "adapter",
    )
    manifest = build_strategy_manifest(
        _Model(), config, paths, (10, 100, 0.1), {"cuda_available": True}
    )
    assert manifest["strategy"] == "lora"
    assert manifest["adapter_based"] is True
    assert manifest["quantized_base"] is False
    assert manifest["actual_dtype"] == "bfloat16"
    assert manifest["matched_modules"] == [
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.0.self_attn.v_proj",
    ]
