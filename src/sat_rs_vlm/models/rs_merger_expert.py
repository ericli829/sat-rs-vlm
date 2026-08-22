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


def rs_detail_parameter_count(
    visual_hidden_size: int,
    llm_hidden_size: int,
    detail_hidden_size: int,
    *,
    local_depth: int = 1,
    spatial_merge_size: int = 2,
) -> int:
    """Return the exact trainable parameter count for one RS detail tap."""

    visual = int(visual_hidden_size)
    llm = int(llm_hidden_size)
    detail = int(detail_hidden_size)
    merge = int(spatial_merge_size)
    depth = int(local_depth)
    if min(visual, llm, detail, merge) <= 0:
        raise ValueError("All RS detail dimensions must be positive")
    if depth not in {1, 2}:
        raise ValueError("local_depth must be 1 or 2")
    layer_norm = 2 * visual
    down = visual * detail + detail
    depthwise = depth * (detail * 3 * 3 + detail)
    pointwise = depth * (detail * detail + detail)
    output = (detail * merge**2) * llm + llm
    return layer_norm + down + depthwise + pointwise + output


class RSDetailResidualBranch(nn.Module):
    """Independent pre-compression local-detail residual for one ViT tap."""

    def __init__(
        self,
        visual_hidden_size: int,
        llm_hidden_size: int,
        *,
        detail_hidden_size: int = 512,
        local_depth: int = 1,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.visual_hidden_size = int(visual_hidden_size)
        self.llm_hidden_size = int(llm_hidden_size)
        self.detail_hidden_size = int(detail_hidden_size)
        self.local_depth = int(local_depth)
        if self.local_depth not in {1, 2}:
            raise ValueError("local_depth must be 1 or 2")
        self.spatial_merge_size = int(spatial_merge_size)
        self.norm = nn.LayerNorm(self.visual_hidden_size)
        self.down = nn.Linear(self.visual_hidden_size, self.detail_hidden_size)
        # Keep the historical D1 state keys exactly stable for C2/C3 continuation.
        self.local_mix = RSLocalMix(self.detail_hidden_size)
        self.extra_local_mix = nn.ModuleList(
            RSLocalMix(self.detail_hidden_size) for _ in range(self.local_depth - 1)
        )
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
            nchw = self.local_mix(nchw)
            for block in self.extra_local_mix:
                nchw = block(nchw)
            mixed.append(nchw.permute(0, 2, 3, 1).contiguous())
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
        self._raw_features: Tensor | None = None
        if variant == "rs_detail" and detail_branch is None:
            raise ValueError("rs_detail merger requires a detail branch")

    def set_grid(self, grid_thw: Tensor | Sequence[Sequence[int]]) -> None:
        self._grid_thw = grid_thw

    def clear_raw_features(self) -> None:
        self._raw_features = None

    def clear_runtime_state(self) -> None:
        """Drop per-forward tensor references retained by hooks/routes."""

        self._raw_features = None
        self._grid_thw = None

    def set_raw_features(self, raw_features: Tensor) -> None:
        if raw_features.ndim != 2:
            raise RuntimeError(
                f"ViT tap must be a raw [tokens, hidden] tensor, got {tuple(raw_features.shape)}"
            )
        self._raw_features = raw_features

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
        raw_features = self._raw_features
        self._raw_features = None
        if raw_features is None:
            raise RuntimeError("RS detail route did not capture its raw ViT block output")
        if raw_features.shape[-1] != self.detail_branch.visual_hidden_size:
            raise RuntimeError(
                "RS detail route received a non-raw ViT feature width: "
                f"expected={self.detail_branch.visual_hidden_size}, "
                f"actual={raw_features.shape[-1]}"
            )
        base_output = self.base(hidden_states)
        delta = self.detail_branch(raw_features, self._grid_thw)
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
        nested_base = None
        get_base_layer = getattr(base, "get_base_layer", None)
        if callable(get_base_layer):
            nested_base = get_base_layer()
        if nested_base is None:
            nested_base = getattr(base, "base_layer", None)
        compute_dtype = None
        for candidate in (nested_base, base):
            candidate_dtype = getattr(candidate, "compute_dtype", None)
            if isinstance(candidate_dtype, torch.dtype) and (
                candidate_dtype.is_floating_point or candidate_dtype.is_complex
            ):
                compute_dtype = candidate_dtype
                break
        if compute_dtype is None:
            compute_dtype = reference.dtype
        if not (compute_dtype.is_floating_point or compute_dtype.is_complex):
            raise TypeError(
                "Quantized interface LoRA target must expose a floating compute_dtype; "
                f"got storage dtype={reference.dtype}, compute_dtype={compute_dtype}"
            )
        self.lora_A.to(device=reference.device, dtype=compute_dtype)
        self.lora_B.to(device=reference.device, dtype=compute_dtype)
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


class CountAuxiliaryHead(nn.Module):
    """Question-conditioned count head stored with the merger expert sidecar."""

    def __init__(
        self,
        llm_hidden_size: int,
        *,
        head_hidden_size: int = 512,
        max_count: int,
        distribution: str = "categorical",
    ) -> None:
        super().__init__()
        if max_count < 1:
            raise ValueError("max_count must be positive")
        if distribution not in {"categorical", "negative_binomial"}:
            raise ValueError(f"Unsupported count distribution: {distribution}")
        self.max_count = int(max_count)
        self.distribution = distribution
        output_size = self.max_count + 1 if distribution == "categorical" else 2
        self.network = nn.Sequential(
            nn.LayerNorm(int(llm_hidden_size)),
            nn.Linear(int(llm_hidden_size), int(head_hidden_size)),
            nn.GELU(),
            nn.Linear(int(head_hidden_size), output_size),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.network(hidden_states)


class RSMergerExpertController:
    """Lifecycle owner for instance-local visual and language route wrappers."""

    def __init__(
        self,
        model: nn.Module,
        *,
        variant: str,
        interface_lora_enabled: bool = False,
        detail_hidden_size: int = 512,
        local_depth: int = 1,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> None:
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported expert variant: {variant!r}")
        self.model = model
        self.variant = variant
        self.detail_hidden_size = int(detail_hidden_size)
        self.local_depth = int(local_depth)
        self.count_head: nn.Module | None = None
        self.count_head_distribution: str | None = None
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
                    local_depth=local_depth,
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
        self._tap_hooks: list[Any] = []
        if variant == "rs_detail":
            blocks = list(self.visual.blocks)
            deepstack_indexes = list(
                getattr(
                    self.visual,
                    "deepstack_visual_indexes",
                    getattr(config, "deepstack_visual_indexes", []),
                )
            )
            tap_indexes = [*deepstack_indexes, len(blocks) - 1]
            if len(deepstack_indexes) != 3 or len(set(tap_indexes)) != 4:
                raise ValueError(
                    "RS detail route requires three distinct DeepStack taps plus the final block"
                )
            if min(tap_indexes) < 0 or max(tap_indexes) >= len(blocks):
                raise ValueError(f"ViT tap indexes are out of range: {tap_indexes}")
            for merger_index, block_index in enumerate(tap_indexes):
                self._tap_hooks.append(
                    blocks[block_index].register_forward_hook(
                        self._capture_raw_tap(merger_index, block_index)
                    )
                )
        self.interface_paths: list[str] = []
        self.interface_modules: list[RoutedInterfaceLoRALinear] = []
        self._interface_bindings: list[tuple[nn.Module, str, nn.Module]] = []
        if interface_lora_enabled:
            self._attach_interface_lora(lora_rank, lora_alpha, lora_dropout)
        self._closed = False

    def configure_count_head(
        self,
        *,
        max_count: int,
        head_hidden_size: int = 512,
        distribution: str = "categorical",
    ) -> nn.Module:
        """Attach an opt-in auxiliary head without modifying the foundation model."""

        if self.count_head is not None:
            raise RuntimeError("Count head is already configured")
        _language_path, layers = _find_language_layers(self.model)
        first_attention = layers[0].self_attn
        projection = first_attention.q_proj
        while isinstance(projection, RoutedInterfaceLoRALinear):
            projection = projection.base
        llm_hidden_size = int(projection.in_features)
        reference = next(self.model.parameters())
        head = CountAuxiliaryHead(
            llm_hidden_size,
            head_hidden_size=head_hidden_size,
            max_count=max_count,
            distribution=distribution,
        )
        head.to(device=reference.device, dtype=reference.dtype)
        self.count_head = head
        self.count_head_distribution = distribution
        return head

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
            merger.clear_raw_features()

    def _capture_raw_tap(self, merger_index: int, block_index: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            raw_features = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(raw_features, Tensor):
                raise RuntimeError(
                    f"ViT block {block_index} did not return a tensor as its first output"
                )
            self.routed_mergers[merger_index].set_raw_features(raw_features)

        return hook

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
        count_head_names: list[str] = []
        count_head_count = 0
        if self.count_head is not None:
            for name, parameter in self.count_head.named_parameters():
                parameter.requires_grad = True
                count_head_names.append(f"count_head.{name}")
                count_head_count += int(parameter.numel())
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
        if self.count_head is not None:
            allowed_ids.update(id(parameter) for parameter in self.count_head.parameters())
        unexpected = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed_ids
        ]
        if unexpected:
            raise AssertionError(f"Unexpected trainable parameters: {unexpected}")
        model_total_count = sum(int(parameter.numel()) for parameter in self.model.parameters())
        total_count = model_total_count + count_head_count
        trainable_count = expert_count + lora_count + count_head_count
        expected_expert_count: int | None = None
        if self.variant == "rs_detail":
            branch = self.routed_mergers[0].detail_branch
            assert branch is not None
            expected_expert_count = len(self.routed_mergers) * rs_detail_parameter_count(
                branch.visual_hidden_size,
                branch.llm_hidden_size,
                branch.detail_hidden_size,
                local_depth=branch.local_depth,
                spatial_merge_size=branch.spatial_merge_size,
            )
            if expert_count != expected_expert_count:
                raise AssertionError(
                    "RS detail parameter formula mismatch: "
                    f"expected={expected_expert_count}, actual={expert_count}"
                )
        return {
            "expert_parameter_count": expert_count,
            "expert_per_tap_parameter_count": (
                expert_count // len(self.routed_mergers) if self.routed_mergers else 0
            ),
            "interface_lora_parameter_count": lora_count,
            "count_head_parameter_count": count_head_count,
            "total_trainable_parameter_count": trainable_count,
            "base_parameter_count": total_count - trainable_count,
            "total_parameter_count": total_count,
            "expert_names": expert_names,
            "interface_lora_names": lora_names,
            "count_head_names": count_head_names,
            "unexpected_trainable": unexpected,
            "expert_variant": self.variant,
            "detail_hidden_size": self.detail_hidden_size,
            "local_depth": self.local_depth,
            "count_head_distribution": self.count_head_distribution,
            "expert_tap_count": len(self.routed_mergers),
            "expected_expert_parameter_count": expected_expert_count,
        }

    def set_training_mode(self, training: bool = True) -> None:
        """Keep the frozen foundation in eval while enabling only expert stochastic layers."""

        self.model.eval()
        if not training:
            return
        for merger in self.routed_mergers:
            expert = merger.clone if self.variant == "clone" else merger.detail_branch
            if expert is not None:
                expert.train(True)
        for module in self.interface_modules:
            # Calling module.train() would recursively re-enable its frozen base projection.
            module.training = True
            module.base.eval()
            module.dropout.train(True)
            module.lora_A.train(True)
            module.lora_B.train(True)
        if self.count_head is not None:
            self.count_head.train(True)

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
        if self.count_head is not None:
            for name, value in self.count_head.state_dict().items():
                state[f"count_head.{name}"] = value
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
        if self.count_head is not None:
            prefix = "count_head."
            self.count_head.load_state_dict(
                {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                },
                strict=True,
            )

    def close(self, *, restore_modules: bool = True) -> None:
        """Remove hooks and optionally restore the exact pre-controller model modules."""

        if self._closed:
            return
        for merger in self.routed_mergers:
            merger.clear_runtime_state()
        if self._pre_hook is not None:
            self._pre_hook.remove()
            self._pre_hook = None
        for handle in self._tap_hooks:
            handle.remove()
        self._tap_hooks.clear()
        if restore_modules:
            self.visual.merger = self._base_main_merger
            self.visual.deepstack_merger_list = nn.ModuleList(self._base_deepstack_mergers)
            for parent, target, base in self._interface_bindings:
                setattr(parent, target, base)
        self._interface_bindings.clear()
        self.interface_modules.clear()
        self.interface_paths.clear()
        self.count_head = None
        self.count_head_distribution = None
        self._closed = True


def merger_description(
    module: nn.Module,
    path: str,
    *,
    visual_hidden_size: int,
    llm_hidden_size: int,
    spatial_merge_size: int,
    expected_postshuffle_norm: bool,
) -> dict[str, Any]:
    if isinstance(module, RoutedMerger):
        module = module.base
    norm = getattr(module, "norm", None)
    fc1 = getattr(module, "linear_fc1", getattr(module, "fc1", None))
    fc2 = getattr(module, "linear_fc2", getattr(module, "fc2", None))
    if norm is None or fc1 is None or fc2 is None:
        raise ValueError(f"Merger {path} does not expose norm/linear_fc1/linear_fc2")
    packed_size = visual_hidden_size * spatial_merge_size**2
    expected_norm_shape = [packed_size if expected_postshuffle_norm else visual_hidden_size]
    actual_norm_shape = list(getattr(norm, "normalized_shape", ()))
    actual_postshuffle_norm = bool(getattr(module, "use_postshuffle_norm", False))
    expected_fc1_shape = [packed_size, packed_size]
    expected_fc2_shape = [llm_hidden_size, packed_size]
    mismatches = []
    if actual_postshuffle_norm != expected_postshuffle_norm:
        mismatches.append(
            "use_postshuffle_norm: "
            f"expected={expected_postshuffle_norm}, actual={actual_postshuffle_norm}"
        )
    if actual_norm_shape != expected_norm_shape:
        mismatches.append(f"norm_shape: expected={expected_norm_shape}, actual={actual_norm_shape}")
    if int(getattr(module, "hidden_size", -1)) != packed_size:
        mismatches.append(
            f"hidden_size: expected={packed_size}, actual={getattr(module, 'hidden_size', None)}"
        )
    if list(fc1.weight.shape) != expected_fc1_shape:
        mismatches.append(
            f"fc1_shape: expected={expected_fc1_shape}, actual={list(fc1.weight.shape)}"
        )
    if list(fc2.weight.shape) != expected_fc2_shape:
        mismatches.append(
            f"fc2_shape: expected={expected_fc2_shape}, actual={list(fc2.weight.shape)}"
        )
    if mismatches:
        raise ValueError(f"Merger {path} architecture mismatch: " + "; ".join(mismatches))

    reference = fc1.weight
    raw_probe = torch.zeros(
        spatial_merge_size**2,
        visual_hidden_size,
        device=reference.device,
        dtype=reference.dtype,
    )
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            probe_output = module(raw_probe)
    except Exception as exc:
        raise ValueError(
            f"Merger {path} rejected raw [{spatial_merge_size**2}, {visual_hidden_size}] input"
        ) from exc
    finally:
        module.train(was_training)
    expected_probe_shape = (1, llm_hidden_size)
    if tuple(probe_output.shape) != expected_probe_shape:
        raise ValueError(
            f"Merger {path} raw-input probe output mismatch: "
            f"expected={expected_probe_shape}, actual={tuple(probe_output.shape)}"
        )
    return {
        "path": path,
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "forward_input_shape_probe": list(raw_probe.shape),
        "forward_input_feature_size": visual_hidden_size,
        "use_postshuffle_norm": actual_postshuffle_norm,
        "norm_shape": actual_norm_shape,
        "fc1_shape": list(fc1.weight.shape),
        "fc2_shape": list(fc2.weight.shape),
        "forward_output_shape_probe": list(probe_output.shape),
        "raw_input_contract_passed": True,
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
    visual_hidden_size = int(config.hidden_size)
    llm_hidden_size = int(config.out_hidden_size)
    spatial_merge_size = int(visual.spatial_merge_size)
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
    merger_kwargs = {
        "visual_hidden_size": visual_hidden_size,
        "llm_hidden_size": llm_hidden_size,
        "spatial_merge_size": spatial_merge_size,
    }
    mergers = [
        merger_description(
            visual.merger,
            f"{visual_path}.merger",
            expected_postshuffle_norm=False,
            **merger_kwargs,
        )
    ]
    mergers.extend(
        merger_description(
            module,
            f"{visual_path}.deepstack_merger_list.{index}",
            expected_postshuffle_norm=True,
            **merger_kwargs,
        )
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
        "vision_hidden_size": visual_hidden_size,
        "llm_hidden_size": llm_hidden_size,
        "spatial_merge_size": spatial_merge_size,
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
