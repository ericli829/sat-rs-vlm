from __future__ import annotations

import pytest

from sat_rs_vlm.training.config import (
    OptimizationGroupConfig,
    TrainableAuditConfig,
    VisionTuningConfig,
)
from sat_rs_vlm.training.optimizer import build_h1_parameter_groups
from sat_rs_vlm.training.vision_tuning import (
    VISUAL_SIDECAR_FILENAME,
    configure_h1_trainable_parameters,
    load_visual_sidecar,
    save_visual_sidecar,
)

torch = pytest.importorskip("torch")
nn = torch.nn


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.merger = nn.Linear(4, 4)
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])


class FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = FakeVisual()
        self.language_model = nn.Linear(4, 4)


class FakeConditional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeCore()


class FakePeftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = FakeConditional()
        self.lora_A = nn.Parameter(torch.ones(4, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 4))
        self.unexpected_head = nn.Linear(4, 4)


def test_only_last_n_blocks_merger_and_lora_are_trainable() -> None:
    model = FakePeftModel()
    audit = configure_h1_trainable_parameters(
        model,
        VisionTuningConfig(
            enabled=True,
            unfreeze_last_n_blocks=2,
            train_main_merger=True,
            train_deepstack_mergers=False,
            train_patch_embed=False,
        ),
        TrainableAuditConfig(fail_on_unexpected_trainable=True),
    )
    visual = model.base_model.model.model.visual

    assert all(not parameter.requires_grad for parameter in visual.blocks[0].parameters())
    assert all(not parameter.requires_grad for parameter in visual.blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in visual.blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in visual.blocks[3].parameters())
    assert all(parameter.requires_grad for parameter in visual.merger.parameters())
    assert all(not parameter.requires_grad for parameter in visual.patch_embed.parameters())
    assert all(
        not parameter.requires_grad for parameter in visual.deepstack_merger_list.parameters()
    )
    assert model.lora_A.requires_grad and model.lora_B.requires_grad
    assert all(not parameter.requires_grad for parameter in model.unexpected_head.parameters())
    assert audit["vision_blocks"]["block_indices"] == [2, 3]
    assert audit["other_trainable"] == []


def test_optimizer_groups_have_distinct_lrs_and_no_duplicate_parameters() -> None:
    model = FakePeftModel()
    audit = configure_h1_trainable_parameters(
        model,
        VisionTuningConfig(enabled=True, unfreeze_last_n_blocks=2),
        TrainableAuditConfig(),
    )
    rates = OptimizationGroupConfig(
        lora_lr=1.0e-5,
        visual_merger_lr=5.0e-6,
        vision_lr=1.0e-6,
    )
    groups = build_h1_parameter_groups(model, audit, rates, weight_decay=0.01)

    assert {group["group_name"]: group["lr"] for group in groups} == {
        "lora": 1.0e-5,
        "visual_merger": 5.0e-6,
        "vision_blocks": 1.0e-6,
    }
    identities = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identities) == len(set(identities))
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert set(identities) == trainable


def test_requesting_more_blocks_than_available_fails() -> None:
    with pytest.raises(ValueError, match="exceeds available visual blocks"):
        configure_h1_trainable_parameters(
            FakePeftModel(),
            VisionTuningConfig(enabled=True, unfreeze_last_n_blocks=5),
            TrainableAuditConfig(),
        )


def test_visual_sidecar_round_trip_uses_generic_name(tmp_path) -> None:
    pytest.importorskip("safetensors")
    model = FakePeftModel()
    audit = configure_h1_trainable_parameters(
        model,
        VisionTuningConfig(enabled=True, unfreeze_last_n_blocks=2),
        TrainableAuditConfig(),
    )
    expected = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in audit["vision_blocks"]["names"] + audit["visual_merger"]["names"]
    }
    saved = save_visual_sidecar(model, audit, tmp_path)
    assert saved.name == VISUAL_SIDECAR_FILENAME
    assert (tmp_path / "visual_trainable_manifest.json").is_file()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in expected:
                parameter.zero_()
    loaded = load_visual_sidecar(model, tmp_path)
    assert set(loaded) == set(expected)
    for name, parameter in model.named_parameters():
        if name in expected:
            assert torch.equal(parameter, expected[name])
