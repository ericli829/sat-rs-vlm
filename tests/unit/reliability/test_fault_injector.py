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
