from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sat_rs_vlm.training.model_audit import (
    audit_lora_targets,
    finalize_lora_trainable_audit,
    model_fingerprint,
    validate_adapter_architecture,
)


class FakeModel:
    def __init__(self, hidden_size: int = 2048) -> None:
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            architectures=["Qwen3VLForConditionalGeneration"],
            text_config=SimpleNamespace(
                hidden_size=hidden_size,
                num_hidden_layers=36,
                num_attention_heads=16,
                vocab_size=151936,
            ),
            vision_config=SimpleNamespace(hidden_size=1152, depth=32),
        )

    def named_modules(self):  # type: ignore[no-untyped-def]
        yield "model.layers.0.self_attn.q_proj", object()
        yield "model.layers.0.self_attn.k_proj", object()


class FakeParameter:
    def __init__(self, count: int, trainable: bool) -> None:
        self._count = count
        self.requires_grad = trainable

    def numel(self) -> int:
        return self._count


def test_target_audit_requires_every_configured_target() -> None:
    report = audit_lora_targets(FakeModel(), ["q_proj", "k_proj"])
    assert report["target_match_counts"] == {"q_proj": 1, "k_proj": 1}
    with pytest.raises(ValueError, match="v_proj"):
        audit_lora_targets(FakeModel(), ["q_proj", "v_proj"])


def test_trainable_audit_attributes_lora_parameters_to_targets() -> None:
    model = FakeModel()
    model.named_parameters = lambda: iter(  # type: ignore[attr-defined,method-assign]
        [
            ("base.q_proj.weight", FakeParameter(100, False)),
            ("base.q_proj.lora_A.default.weight", FakeParameter(8, True)),
            ("base.k_proj.lora_B.default.weight", FakeParameter(12, True)),
        ]
    )
    report = finalize_lora_trainable_audit(
        model,
        {"target_match_counts": {"q_proj": 1, "k_proj": 1}},
    )
    assert report["trainable_parameters_by_target"] == {"q_proj": 8, "k_proj": 12}
    assert report["trainable_parameters"] == 20


def test_adapter_hidden_size_mismatch_fails_before_peft_load(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    fingerprint = model_fingerprint(FakeModel(hidden_size=1536))
    (adapter / "strategy_manifest.json").write_text(
        json.dumps({"base_model_fingerprint": fingerprint}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_adapter_architecture(
            FakeModel(hidden_size=2048), adapter, require_fingerprint=True
        )


def test_strict_cycle_rejects_legacy_adapter_without_fingerprint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lacks base_model_fingerprint"):
        validate_adapter_architecture(FakeModel(), tmp_path, require_fingerprint=True)
