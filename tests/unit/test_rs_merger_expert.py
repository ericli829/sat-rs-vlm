from __future__ import annotations

import json
import time

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from sat_rs_vlm.models.rs_merger_expert import (  # noqa: E402
    BASE_EXPERT,
    COUNTING_EXPERT,
    RSDetailResidualBranch,
    RSMergerExpertController,
    repack_qwen_merge_order,
    unpack_to_spatial_grid,
)


class ToyMerger(nn.Module):
    def __init__(self, hidden: int = 8, output: int = 12, merge: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden * merge**2
        self.norm = nn.LayerNorm(hidden)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, output)

    def forward(self, values):
        values = self.norm(values).reshape(-1, self.hidden_size)
        return self.linear_fc2(torch.nn.functional.gelu(self.linear_fc1(values)))


class ToyAttention(nn.Module):
    def __init__(self, hidden: int = 12) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = ToyAttention()


class ToyLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer() for _ in range(6)])


class ToyVisionConfig:
    hidden_size = 8
    out_hidden_size = 12
    spatial_merge_size = 2
    deepstack_visual_indexes = [0, 1, 2]


class ToyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ToyVisionConfig()
        self.spatial_merge_size = 2
        self.patch_embed = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.deepstack_merger_list = nn.ModuleList([ToyMerger() for _ in range(3)])
        self.merger = ToyMerger()

    def forward(self, hidden_states, grid_thw=None):
        deep = [merger(hidden_states) for merger in self.deepstack_merger_list]
        return self.merger(hidden_states), deep


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = ToyVisual()
        self.language_model = ToyLanguage()


@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[1, 4, 4], [2, 6, 4]], [[3, 8, 10]]])
def test_spatial_round_trip_preserves_qwen_block_major_order(grid):
    token_count = sum(t * h * w for t, h, w in grid)
    values = torch.arange(token_count * 3).reshape(token_count, 3)
    unpacked = unpack_to_spatial_grid(values, grid, spatial_merge_size=2)
    restored = repack_qwen_merge_order(unpacked, spatial_merge_size=2)
    assert torch.equal(restored, values)


@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[1, 4, 4], [2, 6, 4]]])
def test_detail_branch_zero_init_and_token_budget(grid):
    count = sum(t * h * w for t, h, w in grid)
    branch = RSDetailResidualBranch(8, 12, detail_hidden_size=4, spatial_merge_size=2)
    output = branch(torch.randn(count, 8), grid)
    assert output.shape == (count // 4, 12)
    assert torch.count_nonzero(output).item() == 0
    assert branch.output.weight.grad is None


def test_c1_clone_init_and_base_route_isolation():
    torch.manual_seed(7)
    model = ToyModel()
    controller = RSMergerExpertController(model, variant="clone")
    values = torch.randn(24, 8)
    grid = torch.tensor([[1, 4, 6]])
    controller.set_active_expert(BASE_EXPERT)
    base, _ = model.visual(values, grid_thw=grid)
    controller.set_active_expert(COUNTING_EXPERT)
    clone, _ = model.visual(values, grid_thw=grid)
    assert torch.equal(base, clone)
    with torch.no_grad():
        controller.routed_mergers[-1].clone.linear_fc2.bias.add_(1)
    changed, _ = model.visual(values, grid_thw=grid)
    assert not torch.equal(base, changed)
    controller.set_active_expert(BASE_EXPERT)
    base_after, _ = model.visual(values, grid_thw=grid)
    assert torch.equal(base, base_after)


def test_c2_step0_all_four_taps_equal_base():
    torch.manual_seed(11)
    model = ToyModel()
    controller = RSMergerExpertController(model, variant="rs_detail", detail_hidden_size=4)
    values = torch.randn(24, 8)
    grid = torch.tensor([[1, 4, 6]])
    controller.set_active_expert(BASE_EXPERT)
    base_final, base_deep = model.visual(values, grid_thw=grid)
    controller.set_active_expert(COUNTING_EXPERT)
    expert_final, expert_deep = model.visual(values, grid_thw=grid)
    assert torch.equal(base_final, expert_final)
    assert all(torch.equal(left, right) for left, right in zip(base_deep, expert_deep, strict=True))


@pytest.mark.parametrize(
    ("variant", "interface", "expected_prefixes"),
    [
        ("clone", False, ("expert_mergers",)),
        ("rs_detail", False, ("expert_mergers",)),
        ("rs_detail", True, ("expert_mergers", "language_model.layers")),
    ],
)
def test_trainable_audit_has_only_declared_surfaces(variant, interface, expected_prefixes):
    model = ToyModel()
    controller = RSMergerExpertController(
        model,
        variant=variant,
        interface_lora_enabled=interface,
        detail_hidden_size=4,
    )
    audit = controller.freeze_base_and_enable_expert()
    assert audit["unexpected_trainable"] == []
    assert audit["expert_parameter_count"] > 0
    if interface:
        assert audit["interface_lora_parameter_count"] == 16 * (12 * 16 + 16 * 12)
        assert len(audit["interface_lora_names"]) == 32
        assert all(
            "layers.0." in name or "layers.1." in name or "layers.2." in name or "layers.3." in name
            for name in audit["interface_lora_names"]
        )
        assert all(
            any(f".{target}." in name for target in ("q_proj", "k_proj", "v_proj", "o_proj"))
            for name in audit["interface_lora_names"]
        )
    else:
        assert audit["interface_lora_parameter_count"] == 0


def test_c3_zero_lora_delta_and_layer_four_is_frozen():
    torch.manual_seed(13)
    model = ToyModel()
    controller = RSMergerExpertController(
        model,
        variant="rs_detail",
        interface_lora_enabled=True,
        detail_hidden_size=4,
    )
    wrapper = model.language_model.layers[0].self_attn.q_proj
    values = torch.randn(2, 3, 12)
    controller.set_active_expert(BASE_EXPERT)
    base = wrapper(values)
    controller.set_active_expert(COUNTING_EXPERT)
    expert = wrapper(values)
    assert torch.equal(base, expert)
    audit = controller.freeze_base_and_enable_expert()
    assert audit["unexpected_trainable"] == []
    assert not any(
        parameter.requires_grad for parameter in model.language_model.layers[4].parameters()
    )


def test_controller_close_restores_original_modules():
    model = ToyModel()
    main = model.visual.merger
    deepstack = list(model.visual.deepstack_merger_list)
    q_proj = model.language_model.layers[0].self_attn.q_proj
    controller = RSMergerExpertController(
        model,
        variant="rs_detail",
        interface_lora_enabled=True,
        detail_hidden_size=4,
    )
    controller.close()
    assert model.visual.merger is main
    assert all(
        current is expected
        for current, expected in zip(model.visual.deepstack_merger_list, deepstack, strict=True)
    )
    assert model.language_model.layers[0].self_attn.q_proj is q_proj


@pytest.mark.gpu
def test_two_step_cuda_smoke_updates_detail_and_interface_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(17)
    model = ToyModel()
    controller = RSMergerExpertController(
        model,
        variant="rs_detail",
        interface_lora_enabled=True,
        detail_hidden_size=4,
    )
    model.cuda()
    audit = controller.freeze_base_and_enable_expert()
    controller.set_active_expert(COUNTING_EXPERT)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    frozen_before = {
        "vit": model.visual.patch_embed.weight.detach().clone(),
        "base_merger": controller.routed_mergers[-1].base.linear_fc2.weight.detach().clone(),
        "llm_layer4": model.language_model.layers[4].self_attn.q_proj.weight.detach().clone(),
    }
    detail_before = controller.routed_mergers[-1].detail_branch.output.weight.detach().clone()
    lora_before = model.language_model.layers[0].self_attn.q_proj.lora_B.weight.detach().clone()
    grid = torch.tensor([[1, 4, 6]], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    grad_norm = None
    for _ in range(2):
        visual_input = torch.randn(24, 8, device="cuda")
        language_input = torch.randn(2, 3, 12, device="cuda")
        final, deep = model.visual(visual_input, grid_thw=grid)
        interface = model.language_model.layers[0].self_attn.q_proj(language_input)
        loss = (
            final.square().mean()
            + sum(value.square().mean() for value in deep)
            + interface.square().mean()
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in trainable
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert audit["unexpected_trainable"] == []
    assert not torch.equal(detail_before, controller.routed_mergers[-1].detail_branch.output.weight)
    assert not torch.equal(
        lora_before, model.language_model.layers[0].self_attn.q_proj.lora_B.weight
    )
    assert torch.equal(frozen_before["vit"], model.visual.patch_embed.weight)
    assert torch.equal(
        frozen_before["base_merger"], controller.routed_mergers[-1].base.linear_fc2.weight
    )
    assert torch.equal(
        frozen_before["llm_layer4"], model.language_model.layers[4].self_attn.q_proj.weight
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "smoke": "toy_component_two_step_cuda",
                "steps": 2,
                "loss": float(loss.detach().item()),
                "grad_norm": grad_norm,
                "step_time_seconds": elapsed / 2,
                "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "device": torch.cuda.get_device_name(0),
            }
        )
    )
