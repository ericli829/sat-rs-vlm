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
    merger_description,
    repack_qwen_merge_order,
    unpack_to_spatial_grid,
)


class ToyMerger(nn.Module):
    def __init__(
        self,
        hidden: int = 8,
        output: int = 12,
        merge: int = 2,
        *,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden * merge**2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else hidden)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, output)
        self.dropout = nn.Dropout(0.2)

    def forward(self, values):
        if self.use_postshuffle_norm:
            values = self.norm(values.reshape(-1, self.hidden_size))
        else:
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
        self.dropout = nn.Dropout(0.2)


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
        self.deepstack_visual_indexes = [0, 1, 2]
        self.deepstack_merger_list = nn.ModuleList(
            [ToyMerger(use_postshuffle_norm=True) for _ in range(3)]
        )
        self.merger = ToyMerger()

    def forward(self, hidden_states, grid_thw=None):
        deep = []
        for block_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if block_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(block_index)
                deep.append(self.deepstack_merger_list[merger_index](hidden_states))
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


class PackedToyMerger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 32
        self.use_postshuffle_norm = True
        self.norm = nn.LayerNorm(32)
        self.linear_fc1 = nn.Linear(32, 32)
        self.linear_fc2 = nn.Linear(32, 12)

    def forward(self, values):
        assert values.shape[-1] == 32
        return self.linear_fc2(torch.nn.functional.gelu(self.linear_fc1(self.norm(values))))


class OffsetBlock(nn.Module):
    def forward(self, values):
        return values + 1


class PackedToyVisual(ToyVisual):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([OffsetBlock() for _ in range(4)])
        self.deepstack_merger_list = nn.ModuleList([PackedToyMerger() for _ in range(3)])
        self.merger = PackedToyMerger()

    def forward(self, hidden_states, grid_thw=None):
        deep = []
        for block_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if block_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(block_index)
                deep.append(self.deepstack_merger_list[merger_index](hidden_states.reshape(-1, 32)))
        return self.merger(hidden_states.reshape(-1, 32)), deep


def test_c2_detail_uses_raw_block_taps_even_when_merger_input_is_packed():
    model = ToyModel()
    model.visual = PackedToyVisual()
    controller = RSMergerExpertController(model, variant="rs_detail", detail_hidden_size=4)
    captured = []
    handles = []
    for merger in controller.routed_mergers:
        handles.append(
            merger.detail_branch.register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach().clone())
            )
        )
    try:
        controller.set_active_expert(COUNTING_EXPERT)
        final, deep = model.visual(torch.zeros(24, 8), grid_thw=torch.tensor([[1, 4, 6]]))
    finally:
        for handle in handles:
            handle.remove()
    assert final.shape == (6, 12)
    assert all(value.shape == (6, 12) for value in deep)
    assert [tuple(value.shape) for value in captured] == [(24, 8)] * 4
    assert [float(value[0, 0]) for value in captured] == [1.0, 2.0, 3.0, 4.0]


def test_merger_audit_checks_raw_input_and_norm_contract():
    report = merger_description(
        ToyMerger(use_postshuffle_norm=True),
        "deepstack.0",
        visual_hidden_size=8,
        llm_hidden_size=12,
        spatial_merge_size=2,
        expected_postshuffle_norm=True,
    )
    assert report["forward_input_shape_probe"] == [4, 8]
    assert report["norm_shape"] == [32]
    assert report["raw_input_contract_passed"] is True
    with pytest.raises(ValueError, match="norm_shape"):
        merger_description(
            ToyMerger(),
            "deepstack.bad",
            visual_hidden_size=8,
            llm_hidden_size=12,
            spatial_merge_size=2,
            expected_postshuffle_norm=True,
        )


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


def test_training_mode_keeps_foundation_eval_and_enables_expert_dropout():
    model = ToyModel()
    controller = RSMergerExpertController(
        model,
        variant="rs_detail",
        interface_lora_enabled=True,
        detail_hidden_size=4,
    )
    controller.freeze_base_and_enable_expert()
    controller.set_training_mode(True)
    assert model.training is False
    assert model.visual.training is False
    assert model.language_model.training is False
    assert model.language_model.dropout.training is False
    assert all(merger.base.training is False for merger in controller.routed_mergers)
    assert all(merger.base.dropout.training is False for merger in controller.routed_mergers)
    assert all(merger.detail_branch.training is True for merger in controller.routed_mergers)
    assert all(module.training is True for module in controller.interface_modules)
    assert all(module.dropout.training is True for module in controller.interface_modules)
    assert all(module.base.training is False for module in controller.interface_modules)

    clone_model = ToyModel()
    clone_controller = RSMergerExpertController(clone_model, variant="clone")
    clone_controller.freeze_base_and_enable_expert()
    clone_controller.set_training_mode(True)
    assert all(merger.base.dropout.training is False for merger in clone_controller.routed_mergers)
    assert all(merger.clone.dropout.training is True for merger in clone_controller.routed_mergers)


def test_frozen_eval_language_keeps_graph_to_detail_branch():
    model = ToyModel()
    controller = RSMergerExpertController(model, variant="rs_detail", detail_hidden_size=4)
    controller.freeze_base_and_enable_expert()
    controller.set_training_mode(True)
    controller.set_active_expert(COUNTING_EXPERT)
    final, _ = model.visual(torch.randn(24, 8), grid_thw=torch.tensor([[1, 4, 6]]))
    frozen_projection = model.language_model.layers[4].self_attn.q_proj
    frozen_projection(final).square().mean().backward()
    assert frozen_projection.weight.grad is None
    assert controller.routed_mergers[-1].detail_branch.output.weight.grad is not None


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
