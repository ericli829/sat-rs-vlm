"""Task-specialized Qwen3-VL merger experts.

This module is deliberately instance-local: it wraps the mergers of one loaded
Qwen model and never monkey-patches a Transformers class.  The base route keeps
the original modules, while the counting route selects either cloned mergers or
zero-initialized remote-sensing detail residuals.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from sat_rs_vlm.training.vision_tuning import resolve_visual_module

BASE_EXPERT = "base"
COUNTING_EXPERT = "counting"
SUPPORTED_EXPERTS = frozenset({BASE_EXPERT, COUNTING_EXPERT})
SUPPORTED_VARIANTS = frozenset({"base", "clone", "rs_detail"})
INTERFACE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
INTERFACE_LAYERS = (0, 1, 2, 3)


@dataclass
class ExpertRouteState:
    """Mutable route shared by merger and shallow-interface wrappers."""

    active_expert: str = BASE_EXPERT

    def set(self, expert: str) -> None:
        if expert not in SUPPORTED_EXPERTS:
            raise ValueError(f"Unsupported active expert: {expert!r}")
        self.active_expert = expert


def route_for_task(task_type: str) -> str:
    """Hard route from canonical benchmark metadata; prompt text is never read."""

    return COUNTING_EXPERT if task_type.strip().lower() == "counting" else BASE_EXPERT


def _normalized_grid_rows(
    image_grid_thw: Tensor | Sequence[Sequence[int]],
) -> list[tuple[int, int, int]]:
    rows = (
        image_grid_thw.detach().cpu().tolist()
        if isinstance(image_grid_thw, Tensor)
        else image_grid_thw
    )
    result: list[tuple[int, int, int]] = []
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"image_grid_thw rows must have three values, got {row!r}")
        t, h, w = (int(value) for value in row)
        if min(t, h, w) <= 0:
            raise ValueError(f"image_grid_thw values must be positive, got {(t, h, w)!r}")
        result.append((t, h, w))
    if not result:
        raise ValueError("image_grid_thw must contain at least one image")
    return result


def unpack_to_spatial_grid(
    qwen_tokens: Tensor,
    image_grid_thw: Tensor | Sequence[Sequence[int]],
    *,
    spatial_merge_size: int,
) -> list[Tensor]:
    """Convert Qwen block-major patch order into per-image raster ``[T,H,W,D]``.

    Qwen's processor orders tokens as ``T,H_block,W_block,H_inner,W_inner`` so
    each spatial merge group is contiguous.  Local convolution must not operate
    on that flattened order; this function restores the actual spatial grid.
    """

    if qwen_tokens.ndim != 2:
        raise ValueError(f"qwen_tokens must be [tokens, hidden], got {tuple(qwen_tokens.shape)}")
    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError("spatial_merge_size must be positive")
    grids: list[Tensor] = []
    offset = 0
    for t, h, w in _normalized_grid_rows(image_grid_thw):
        if h % merge or w % merge:
            raise ValueError(f"Grid {(t, h, w)} is not divisible by spatial_merge_size={merge}")
        count = t * h * w
        stop = offset + count
        if stop > qwen_tokens.shape[0]:
            raise ValueError("image_grid_thw describes more tokens than the input contains")
        block_major = qwen_tokens[offset:stop].reshape(
            t, h // merge, w // merge, merge, merge, qwen_tokens.shape[-1]
        )
        raster = block_major.permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, qwen_tokens.shape[-1])
        grids.append(raster)
        offset = stop
    if offset != qwen_tokens.shape[0]:
        raise ValueError(
            "image_grid_thw token total does not match input: "
            f"grid={offset}, input={qwen_tokens.shape[0]}"
        )
    return grids


def repack_qwen_merge_order(
    spatial_grids: Sequence[Tensor],
    *,
    spatial_merge_size: int,
) -> Tensor:
    """Inverse of :func:`unpack_to_spatial_grid`, preserving Qwen ordering."""

    merge = int(spatial_merge_size)
    packed: list[Tensor] = []
    hidden_size: int | None = None
    for grid in spatial_grids:
        if grid.ndim != 4:
            raise ValueError(f"Spatial grid must be [T,H,W,D], got {tuple(grid.shape)}")
        t, h, w, dim = grid.shape
        if h % merge or w % merge:
            raise ValueError(f"Grid {(t, h, w)} is not divisible by spatial_merge_size={merge}")
        if hidden_size is None:
            hidden_size = dim
        elif hidden_size != dim:
            raise ValueError("All spatial grids must have the same hidden size")
        tokens = grid.reshape(t, h // merge, merge, w // merge, merge, dim)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(t * h * w, dim)
        packed.append(tokens)
    if not packed:
        raise ValueError("At least one spatial grid is required")
    return torch.cat(packed, dim=0)


class RSLocalMix(nn.Module):
    """Depthwise/pointwise local interaction with an internal residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.pointwise(self.activation(self.depthwise(values)))


class RSDetailResidualBranch(nn.Module):
    """Independent pre-compression local-detail residual for one ViT tap."""

    def __init__(
        self,
        visual_hidden_size: int,
        llm_hidden_size: int,
        *,
        detail_hidden_size: int = 512,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.visual_hidden_size = int(visual_hidden_size)
        self.llm_hidden_size = int(llm_hidden_size)
        self.detail_hidden_size = int(detail_hidden_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.norm = nn.LayerNorm(self.visual_hidden_size)
        self.down = nn.Linear(self.visual_hidden_size, self.detail_hidden_size)
        self.local_mix = RSLocalMix(self.detail_hidden_size)
        packed_size = self.detail_hidden_size * self.spatial_merge_size**2
        self.output = nn.Linear(packed_size, self.llm_hidden_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, raw_features: Tensor, image_grid_thw: Tensor | Sequence[Sequence[int]]
    ) -> Tensor:
        projected = self.down(self.norm(raw_features))
        grids = unpack_to_spatial_grid(
            projected,
            image_grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )
        mixed: list[Tensor] = []
        for grid in grids:
            # Treat temporal patches as independent images; the hypothesis is spatial local detail.
            nchw = grid.permute(0, 3, 1, 2).contiguous()
            mixed.append(self.local_mix(nchw).permute(0, 2, 3, 1).contiguous())
        qwen_order = repack_qwen_merge_order(
            mixed,
            spatial_merge_size=self.spatial_merge_size,
        )
        packed = qwen_order.reshape(-1, self.detail_hidden_size * self.spatial_merge_size**2)
        return self.output(packed)


class RoutedMerger(nn.Module):
    """Select base, clone, or base-plus-detail output for one merger tap."""

    def __init__(
        self,
        base: nn.Module,
        route_state: ExpertRouteState,
        *,
        variant: str,
        detail_branch: RSDetailResidualBranch | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.route_state = route_state
        self.variant = variant
        self.clone = copy.deepcopy(base) if variant == "clone" else None
        self.detail_branch = detail_branch
        self._grid_thw: Tensor | Sequence[Sequence[int]] | None = None
        if variant == "rs_detail" and detail_branch is None:
            raise ValueError("rs_detail merger requires a detail branch")

    def set_grid(self, grid_thw: Tensor | Sequence[Sequence[int]]) -> None:
        self._grid_thw = grid_thw

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.route_state.active_expert == BASE_EXPERT or self.variant == "base":
            return self.base(hidden_states)
        if self.variant == "clone":
            assert self.clone is not None
            return self.clone(hidden_states)
        if self._grid_thw is None:
            raise RuntimeError(
                "RS detail route requires image_grid_thw captured by the visual wrapper"
            )
        assert self.detail_branch is not None
        base_output = self.base(hidden_states)
        delta = self.detail_branch(hidden_states, self._grid_thw)
        if delta.shape != base_output.shape:
            delta_shape = tuple(delta.shape)
            base_shape = tuple(base_output.shape)
            raise RuntimeError(
                f"Detail/base output mismatch: delta={delta_shape}, base={base_shape}"
            )
        return base_output + delta

    def expert_named_parameters(self, prefix: str) -> Iterator[tuple[str, nn.Parameter]]:
        module = self.clone if self.variant == "clone" else self.detail_branch
        if module is not None:
            yield from module.named_parameters(prefix=prefix)


class RoutedInterfaceLoRALinear(nn.Module):
    """Exact-path shallow LoRA that is bypassed on the base route."""

    def __init__(
        self,
        base: nn.Module,
        route_state: ExpertRouteState,
        *,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not all(hasattr(base, name) for name in ("in_features", "out_features")):
            raise TypeError(
                "Interface LoRA target must expose in_features/out_features, got "
                f"{type(base).__name__}"
            )
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.route_state = route_state
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Linear(int(base.in_features), self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, int(base.out_features), bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        reference = next(base.parameters())
        self.lora_A.to(device=reference.device, dtype=reference.dtype)
        self.lora_B.to(device=reference.device, dtype=reference.dtype)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, values: Tensor) -> Tensor:
        result = self.base(values)
        if self.route_state.active_expert == COUNTING_EXPERT:
            result = result + self.lora_B(self.lora_A(self.dropout(values))) * self.scaling
        return result


def _find_language_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and layers
            and all(hasattr(layer, "self_attn") for layer in layers)
        ):
            candidates.append((name, layers))
    if not candidates:
        raise ValueError("Could not resolve Qwen language decoder layers")
    # The language stack is the candidate whose attention exposes every q/k/v/o projection.
    exact = [
        item
        for item in candidates
        if all(hasattr(item[1][0].self_attn, target) for target in INTERFACE_TARGETS)
    ]
    if len(exact) != 1:
        names = [name for name, _ in exact or candidates]
        raise ValueError(f"Language decoder layer resolution is ambiguous: {names}")
    return exact[0]


class RSMergerExpertController:
    """Lifecycle owner for instance-local visual and language route wrappers."""

    def __init__(
        self,
        model: nn.Module,
        *,
        variant: str,
        interface_lora_enabled: bool = False,
        detail_hidden_size: int = 512,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> None:
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported expert variant: {variant!r}")
        self.model = model
        self.variant = variant
        self.route_state = ExpertRouteState()
        self.visual = resolve_visual_module(model)
        config = getattr(self.visual, "config", None)
        visual_hidden = int(getattr(config, "hidden_size", 0))
        llm_hidden = int(getattr(config, "out_hidden_size", 0))
        merge = int(
            getattr(config, "spatial_merge_size", getattr(self.visual, "spatial_merge_size", 0))
        )
        if min(visual_hidden, llm_hidden, merge) <= 0:
            raise ValueError("Visual config lacks hidden_size/out_hidden_size/spatial_merge_size")
        deepstack = list(self.visual.deepstack_merger_list)
        if len(deepstack) != 3:
            raise ValueError(
                f"Counting expert requires exactly three DeepStack mergers, got {len(deepstack)}"
            )
        base_mergers = [*deepstack, self.visual.merger]
        self._base_deepstack_mergers = deepstack
        self._base_main_merger = self.visual.merger
        self.routed_mergers: list[RoutedMerger] = []
        for base in base_mergers:
            branch = (
                RSDetailResidualBranch(
                    visual_hidden,
                    llm_hidden,
                    detail_hidden_size=detail_hidden_size,
                    spatial_merge_size=merge,
                )
                if variant == "rs_detail"
                else None
            )
            if branch is not None:
                reference = next(base.parameters())
                branch.to(device=reference.device, dtype=reference.dtype)
            self.routed_mergers.append(
                RoutedMerger(base, self.route_state, variant=variant, detail_branch=branch)
            )
        self.visual.deepstack_merger_list = nn.ModuleList(self.routed_mergers[:3])
        self.visual.merger = self.routed_mergers[3]
        self._pre_hook = self.visual.register_forward_pre_hook(self._capture_grid, with_kwargs=True)
        self.interface_paths: list[str] = []
        self.interface_modules: list[RoutedInterfaceLoRALinear] = []
        self._interface_bindings: list[tuple[nn.Module, str, nn.Module]] = []
        if interface_lora_enabled:
            self._attach_interface_lora(lora_rank, lora_alpha, lora_dropout)

    def _capture_grid(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        grid = kwargs.get("grid_thw", kwargs.get("image_grid_thw"))
        if grid is None and len(args) > 1:
            grid = args[1]
        if grid is None:
            raise RuntimeError("Qwen visual forward did not provide grid_thw")
        for merger in self.routed_mergers:
            merger.set_grid(grid)

    def _attach_interface_lora(self, rank: int, alpha: float, dropout: float) -> None:
        layer_root, layers = _find_language_layers(self.model)
        if len(layers) <= max(INTERFACE_LAYERS):
            raise ValueError(f"Language model has only {len(layers)} layers")
        for layer_index in INTERFACE_LAYERS:
            attention = layers[layer_index].self_attn
            for target in INTERFACE_TARGETS:
                base = getattr(attention, target, None)
                if base is None or not all(
                    hasattr(base, name) for name in ("in_features", "out_features")
                ):
                    raise TypeError(
                        f"{layer_root}.layers.{layer_index}.self_attn.{target} is not linear-like"
                    )
                wrapper = RoutedInterfaceLoRALinear(
                    base,
                    self.route_state,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(attention, target, wrapper)
                self.interface_modules.append(wrapper)
                self._interface_bindings.append((attention, target, base))
                prefix = f"{layer_root}." if layer_root else ""
                self.interface_paths.append(f"{prefix}layers.{layer_index}.self_attn.{target}")

    def set_active_expert(self, expert: str) -> None:
        self.route_state.set(expert)

    def set_task(self, task_type: str) -> None:
        self.set_active_expert(route_for_task(task_type))

    @contextlib.contextmanager
    def activate(self, expert: str) -> Iterator[None]:
        previous = self.route_state.active_expert
        self.set_active_expert(expert)
        try:
            yield
        finally:
            self.set_active_expert(previous)

    def freeze_base_and_enable_expert(self) -> dict[str, Any]:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        expert_names: list[str] = []
        expert_count = 0
        for index, merger in enumerate(self.routed_mergers):
            for name, parameter in merger.expert_named_parameters(f"expert_mergers.{index}"):
                parameter.requires_grad = True
                expert_names.append(name)
                expert_count += int(parameter.numel())
        lora_names: list[str] = []
        lora_count = 0
        for path, module in zip(self.interface_paths, self.interface_modules, strict=True):
            for local_name, parameter in module.named_parameters():
                if not local_name.startswith("lora_"):
                    continue
                parameter.requires_grad = True
                name = f"{path}.{local_name}"
                lora_names.append(name)
                lora_count += int(parameter.numel())
        allowed_ids = {
            id(parameter)
            for merger in self.routed_mergers
            for _, parameter in merger.expert_named_parameters("expert")
        }
        allowed_ids.update(
            id(parameter)
            for module in self.interface_modules
            for name, parameter in module.named_parameters()
            if name.startswith("lora_")
        )
        unexpected = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed_ids
        ]
        if unexpected:
            raise AssertionError(f"Unexpected trainable parameters: {unexpected}")
        total_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        trainable_count = expert_count + lora_count
        return {
            "expert_parameter_count": expert_count,
            "interface_lora_parameter_count": lora_count,
            "total_trainable_parameter_count": trainable_count,
            "base_parameter_count": total_count - trainable_count,
            "total_parameter_count": total_count,
            "expert_names": expert_names,
            "interface_lora_names": lora_names,
            "unexpected_trainable": unexpected,
        }

    def expert_state_dict(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for index, merger in enumerate(self.routed_mergers):
            module = merger.clone if self.variant == "clone" else merger.detail_branch
            if module is not None:
                for name, value in module.state_dict().items():
                    state[f"expert_mergers.{index}.{name}"] = value
        for path, module in zip(self.interface_paths, self.interface_modules, strict=True):
            state[f"interface_lora.{path}.lora_A.weight"] = module.lora_A.weight.detach()
            state[f"interface_lora.{path}.lora_B.weight"] = module.lora_B.weight.detach()
        return state

    def load_expert_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected = set(self.expert_state_dict())
        actual = set(state)
        if expected != actual:
            raise ValueError(
                "Composite expert state keys do not match exactly: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
        for index, merger in enumerate(self.routed_mergers):
            module = merger.clone if self.variant == "clone" else merger.detail_branch
            if module is None:
                continue
            prefix = f"expert_mergers.{index}."
            module.load_state_dict(
                {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                },
                strict=True,
            )
        for path, module in zip(self.interface_paths, self.interface_modules, strict=True):
            module.lora_A.weight.data.copy_(state[f"interface_lora.{path}.lora_A.weight"])
            module.lora_B.weight.data.copy_(state[f"interface_lora.{path}.lora_B.weight"])

    def close(self, *, restore_modules: bool = True) -> None:
        """Remove hooks and optionally restore the exact pre-controller model modules."""

        if self._pre_hook is not None:
            self._pre_hook.remove()
            self._pre_hook = None
        if restore_modules:
            self.visual.merger = self._base_main_merger
            self.visual.deepstack_merger_list = nn.ModuleList(self._base_deepstack_mergers)
            for parent, target, base in self._interface_bindings:
                setattr(parent, target, base)


def merger_description(module: nn.Module, path: str) -> dict[str, Any]:
    norm = getattr(module, "norm", None)
    fc1 = getattr(module, "linear_fc1", getattr(module, "fc1", None))
    fc2 = getattr(module, "linear_fc2", getattr(module, "fc2", None))
    if norm is None or fc1 is None or fc2 is None:
        raise ValueError(f"Merger {path} does not expose norm/linear_fc1/linear_fc2")
    return {
        "path": path,
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "norm_shape": list(getattr(norm, "normalized_shape", ())),
        "fc1_shape": list(fc1.weight.shape),
        "fc2_shape": list(fc2.weight.shape),
        "parameter_count": sum(int(parameter.numel()) for parameter in module.parameters()),
    }


def source_architecture_audit(model: nn.Module) -> dict[str, Any]:
    """Audit the loaded runtime and fail on unresolved injection/module semantics."""

    visual = resolve_visual_module(model)
    module_names = {id(module): name for name, module in model.named_modules()}
    visual_path = module_names.get(id(visual))
    if visual_path is None:
        raise ValueError("Resolved visual module is absent from model.named_modules()")
    config = getattr(visual, "config", None)
    deepstack_indexes = list(
        getattr(visual, "deepstack_visual_indexes", getattr(config, "deepstack_visual_indexes", []))
    )
    deepstack = list(visual.deepstack_merger_list)
    if len(deepstack_indexes) != len(deepstack):
        raise ValueError("DeepStack index/merger count mismatch")
    language_path, layers = _find_language_layers(model)
    source = inspect.getsource(
        type(
            next(module for name, module in model.named_modules() if name == language_path)
        ).forward
    )
    decoder_position = source.find("decoder_layer(")
    injection_position = source.find("_deepstack_process(")
    if decoder_position < 0 or injection_position < 0 or decoder_position >= injection_position:
        raise ValueError("Could not prove that DeepStack injection occurs after each decoder layer")
    mergers = [merger_description(visual.merger, f"{visual_path}.merger")]
    mergers.extend(
        merger_description(module, f"{visual_path}.deepstack_merger_list.{index}")
        for index, module in enumerate(deepstack)
    )
    attention_paths: dict[str, dict[str, str]] = {}
    for layer_index in INTERFACE_LAYERS:
        if layer_index >= len(layers):
            raise ValueError("Language model does not expose decoder layers 0-3")
        paths: dict[str, str] = {}
        for target in INTERFACE_TARGETS:
            module = getattr(layers[layer_index].self_attn, target, None)
            path = module_names.get(id(module))
            if path is None:
                raise ValueError(f"Could not resolve named path for layer {layer_index} {target}")
            paths[target] = path
        attention_paths[str(layer_index)] = paths
    tap_mapping = [
        {
            "vit_block": int(index),
            "deepstack_feature": ds_index,
            "injected_after_llm_layer": ds_index,
            "first_consumed_by_llm_layer": ds_index + 1,
        }
        for ds_index, index in enumerate(deepstack_indexes)
    ]
    tap_mapping.append(
        {
            "vit_block": len(list(visual.blocks)) - 1,
            "deepstack_feature": None,
            "injected_before_llm_layer": 0,
            "first_consumed_by_llm_layer": 0,
        }
    )
    return {
        "schema_version": "1.0",
        "vision_block_count": len(list(visual.blocks)),
        "vision_hidden_size": int(config.hidden_size),
        "llm_hidden_size": int(config.out_hidden_size),
        "spatial_merge_size": int(visual.spatial_merge_size),
        "deepstack_visual_indexes": deepstack_indexes,
        "visual_module_path": visual_path,
        "visual_module_class": f"{visual.__class__.__module__}.{visual.__class__.__qualname__}",
        "mergers": mergers,
        "deepstack_injection_order": tap_mapping,
        "injection_source_proof": (
            "decoder_layer call precedes _deepstack_process in runtime class source"
        ),
        "language_layer_attention_paths": attention_paths,
    }


def validate_expected_qwen4b_contract(audit: Mapping[str, Any]) -> None:
    expected = {
        "vision_block_count": 24,
        "vision_hidden_size": 1024,
        "llm_hidden_size": 2560,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [5, 11, 17],
    }
    mismatches = [
        f"{key}: expected={value!r}, actual={audit.get(key)!r}"
        for key, value in expected.items()
        if audit.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Qwen3-VL-4B architecture mismatch; expert training is blocked: "
            + "; ".join(mismatches)
        )
