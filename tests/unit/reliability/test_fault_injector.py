import json
from pathlib import Path

import pytest

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.fault_injector import (
    ParameterSelector,
    bit_indices_for_tensor,
    fault_bits_from_density,
    inject_model_parameter_bitflips,
    inject_safetensors_adapter,
    inject_state_dict_bitflips,
    load_safetensors_state,
    model_fault_inventory,
    save_safetensors_state,
    selectable_parameters,
    selector_for_fault_target,
    summarize_fault_inventory,
)


def _state_dict() -> dict[str, object]:
    torch = pytest.importorskip("torch")
    return {
        "model.layers.0.q_proj.lora_A.default.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.0.q_proj.lora_B.default.weight": torch.ones(4, dtype=torch.float32),
        "model.layers.1.k_proj.lora_A.default.weight": torch.ones(4, dtype=torch.float16),
        "model.embed.weight": torch.ones(4, dtype=torch.float32),
        "metadata": "not-a-tensor",
    }


def test_lora_a_b_regex_and_layer_selectors() -> None:
    state = _state_dict()

    lora_a = selectable_parameters(state, ParameterSelector(lora_scope="a"))
    lora_b = selectable_parameters(state, ParameterSelector(lora_scope="b"))
    layer_one = selectable_parameters(
        state,
        ParameterSelector(name_regex=r"k_proj", layer_indices=(1,), lora_scope="all"),
    )

    assert [name for name, _ in lora_a] == [
        "model.layers.0.q_proj.lora_A.default.weight",
        "model.layers.1.k_proj.lora_A.default.weight",
    ]
    assert [name for name, _ in lora_b] == ["model.layers.0.q_proj.lora_B.default.weight"]
    assert [name for name, _ in layer_one] == ["model.layers.1.k_proj.lora_A.default.weight"]


def test_state_dict_injection_only_changes_allowed_parameter() -> None:
    torch = pytest.importorskip("torch")
    clean = _state_dict()
    clean_a = clean["model.layers.0.q_proj.lora_A.default.weight"].clone()

    fault, records = inject_state_dict_bitflips(
        clean,
        num_bits=2,
        seed=5,
        selector=ParameterSelector(lora_scope="b"),
    )

    assert len(records) == 2
    assert {record.target_name for record in records} == {
        "model.layers.0.q_proj.lora_B.default.weight"
    }
    assert torch.equal(clean["model.layers.0.q_proj.lora_A.default.weight"], clean_a)
    assert torch.equal(
        fault["model.layers.0.q_proj.lora_A.default.weight"],
        clean["model.layers.0.q_proj.lora_A.default.weight"],
    )


def test_no_matching_parameter_fails() -> None:
    with pytest.raises(ValueError, match="No tensor parameters"):
        inject_state_dict_bitflips(
            _state_dict(),
            num_bits=1,
            seed=1,
            selector=ParameterSelector(name_regex=r"does_not_exist"),
        )


def test_visual_sidecar_regions_are_selectable_by_target_and_layer() -> None:
    torch = pytest.importorskip("torch")
    state = {
        "base_model.model.visual.blocks.25.attn.weight": torch.ones(2),
        "base_model.model.visual.blocks.26.attn.weight": torch.ones(2),
        "base_model.model.visual.merger.mlp.weight": torch.ones(2),
        "base_model.model.model.layers.0.self_attn.weight": torch.ones(2),
    }
    blocks = selectable_parameters(
        state,
        selector_for_fault_target("visual_blocks", layer_indices=(26,)),
    )
    merger = selectable_parameters(state, selector_for_fault_target("visual_merger"))
    assert [name for name, _ in blocks] == ["base_model.model.visual.blocks.26.attn.weight"]
    assert [name for name, _ in merger] == ["base_model.model.visual.merger.mlp.weight"]


def test_named_target_in_memory_injection_changes_only_selected_region() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Linear(2, 2, bias=False)
            self.language = torch.nn.Linear(2, 2, bias=False)

    model = Model()
    clean_visual = model.visual.weight.detach().clone()
    clean_language = model.language.weight.detach().clone()
    records = inject_model_parameter_bitflips(
        model,
        num_bits=1,
        seed=7,
        selector=selector_for_fault_target("vision_encoder"),
    )
    assert len(records) == 1
    assert records[0].target_name.startswith("visual.")
    assert not torch.equal(model.visual.weight, clean_visual)
    assert torch.equal(model.language.weight, clean_language)


def test_unknown_target_and_normalized_density_are_explicit() -> None:
    with pytest.raises(ValueError, match="fault.target"):
        selector_for_fault_target("kv_cache")
    assert fault_bits_from_density(1_000_000, 10) == 10
    assert fault_bits_from_density(30, 1) == 1


def test_float_bit_planes_are_disjoint_and_cover_dtype() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.ones(1, dtype=torch.float32)
    sign = set(bit_indices_for_tensor(tensor, "sign"))
    exponent = set(bit_indices_for_tensor(tensor, "exponent"))
    mantissa = set(bit_indices_for_tensor(tensor, "mantissa"))
    assert not sign & exponent
    assert not sign & mantissa
    assert not exponent & mantissa
    assert sign | exponent | mantissa == set(range(32))


def test_visual_merger_inventory_is_not_misclassified_as_generic_mlp() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.merger = torch.nn.Linear(2, 2)

    inventory = model_fault_inventory(Model())
    groups = summarize_fault_inventory(inventory)
    assert {group["region"] for group in groups} == {"visual_merger"}


def test_safetensors_adapter_injection_preserves_source_and_reloads(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    source = tmp_path / "clean"
    source.mkdir()
    state = _state_dict()
    tensor_state = {name: value for name, value in state.items() if hasattr(value, "dtype")}
    save_safetensors_state(tensor_state, source / "adapter_model.safetensors", {"owner": "test"})
    (source / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}), encoding="utf-8")
    (source / "strategy_manifest.json").write_text("{}", encoding="utf-8")
    processor = source / "processor"
    processor.mkdir()
    (processor / "processor_config.json").write_text("{}", encoding="utf-8")
    historical = source / "checkpoint-500"
    historical.mkdir()
    (historical / "adapter_model.safetensors").write_bytes(b"historical checkpoint")
    (source / "optimizer.pt").write_bytes(b"optimizer state")
    source_hash = file_sha256(source / "adapter_model.safetensors")

    report = inject_safetensors_adapter(
        source,
        tmp_path / "fault",
        num_bits=3,
        seed=9,
        selector=ParameterSelector(lora_scope="a"),
    )
    reloaded, metadata = load_safetensors_state(tmp_path / "fault" / "adapter_model.safetensors")

    assert report.source_unchanged and report.fault_differs and report.reload_verified
    assert file_sha256(source / "adapter_model.safetensors") == source_hash
    assert metadata == {"owner": "test"}
    assert reloaded
    assert (tmp_path / "fault" / "adapter_config.json").is_file()
    assert (tmp_path / "fault" / "fault_records.jsonl").is_file()
    assert (tmp_path / "fault" / "strategy_manifest.json").is_file()
    assert (tmp_path / "fault" / "processor/processor_config.json").is_file()
    assert not (tmp_path / "fault" / "checkpoint-500").exists()
    assert not (tmp_path / "fault" / "optimizer.pt").exists()
