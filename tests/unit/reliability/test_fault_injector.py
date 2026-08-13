import json
from pathlib import Path

import pytest

from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.fault_injector import (
    ParameterSelector,
    inject_safetensors_adapter,
    inject_state_dict_bitflips,
    load_safetensors_state,
    save_safetensors_state,
    selectable_parameters,
    selector_for_fault_target,
    inject_model_parameter_bitflips,
    bit_indices_for_tensor,
    model_fault_inventory,
    fault_bits_from_density,
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


def test_named_fault_targets_and_in_memory_model_injection() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Linear(2, 2, bias=False)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.ModuleDict({"self_attn": torch.nn.Linear(2, 2, bias=False),
                                      "mlp": torch.nn.Linear(2, 2, bias=False)})]
            )

    model = TinyModel()
    clean_visual = model.visual.weight.detach().clone()
    clean_attention = model.model.layers[0]["self_attn"].weight.detach().clone()
    records = inject_model_parameter_bitflips(
        model, num_bits=1, seed=7, selector=selector_for_fault_target("vision_encoder")
    )

    assert len(records) == 1
    assert records[0].target_name.startswith("visual.")
    assert not torch.equal(model.visual.weight, clean_visual)
    assert torch.equal(model.model.layers[0]["self_attn"].weight, clean_attention)
    attention = selector_for_fault_target("attention", layer_indices=(0,))
    assert [name for name, _ in selectable_parameters(dict(model.named_parameters()), attention)] == [
        "model.layers.0.self_attn.weight"
    ]


def test_unknown_fault_target_fails() -> None:
    with pytest.raises(ValueError, match="fault.target"):
        selector_for_fault_target("kv_cache")


def test_bit_plane_indices_and_in_memory_exponent_injection() -> None:
    torch = pytest.importorskip("torch")
    assert bit_indices_for_tensor(torch.zeros(1, dtype=torch.float16), "sign") == (15,)
    assert bit_indices_for_tensor(torch.zeros(1, dtype=torch.float16), "exponent") == (10, 11, 12, 13, 14)
    assert bit_indices_for_tensor(torch.zeros(1, dtype=torch.bfloat16), "mantissa") == tuple(range(7))

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float16)

    records = inject_model_parameter_bitflips(
        TinyModel(), num_bits=3, seed=2, selector=selector_for_fault_target("vision_encoder"),
        bit_plane="exponent",
    )
    assert {record.bit_index for record in records}.issubset({10, 11, 12, 13, 14})


def test_fault_inventory_and_normalized_density() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Linear(2, 3, bias=False).to(dtype=torch.float16)

    inventory = model_fault_inventory(
        TinyModel(), selector=selector_for_fault_target("vision_encoder"), bit_plane="exponent"
    )
    assert inventory["total_elements"] == 6
    assert inventory["candidate_bits"] == 6 * 5
    assert fault_bits_from_density(1_000_000, 10) == 10
    assert fault_bits_from_density(30, 1) == 1


def test_fault_inventory_summary_groups_regions_and_layers() -> None:
    inventory = {
        "parameters": [
            {"name": "model.layers.14.self_attn.q_proj.weight", "elements": 4, "candidate_bits": 64},
            {"name": "model.layers.14.mlp.down_proj.weight", "elements": 8, "candidate_bits": 128},
            {"name": "visual.blocks.2.weight", "elements": 3, "candidate_bits": 48},
            {"name": "model.layers.14.self_attn.q_proj.lora_A.weight", "elements": 2, "candidate_bits": 32},
        ]
    }
    summary = summarize_fault_inventory(inventory)
    by_key = {(row["region"], row["layer"]): row for row in summary}
    assert by_key[("attention", 14)]["candidate_bits"] == 64
    assert by_key[("mlp", 14)]["elements"] == 8
    assert by_key[("vision_encoder", 2)]["tensor_count"] == 1
    assert by_key[("lora_adapter", 14)]["candidate_bits"] == 32
