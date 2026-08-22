"""Question-conditioned auxiliary counting objective for merger experts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from sat_rs_vlm.models.rs_merger_expert import _find_language_layers

IGNORE_COUNT_TARGET = -100


def prompt_anchor_indices(labels: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    """Locate the last prompt token immediately before the first supervised token."""

    if labels.ndim != 2:
        raise ValueError(f"labels must be [batch, sequence], got {tuple(labels.shape)}")
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every sample must contain at least one supervised assistant token")
    first = supervised.to(torch.int64).argmax(dim=1)
    anchors = first - 1
    if bool((anchors < 0).any()):
        raise ValueError("Assistant supervision starts at token zero; no prompt anchor exists")
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask and labels must have identical shapes")
        active = attention_mask.gather(1, anchors[:, None]).squeeze(1).bool()
        if not bool(active.all()):
            raise ValueError("Resolved prompt anchor points to padding")
    return anchors


class EarlyLayerFeatureTap:
    """Capture one decoder layer output through an instance-local forward hook."""

    def __init__(self, model: nn.Module, *, layer_index: int = 3) -> None:
        path, layers = _find_language_layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"Decoder layer index is out of range: {layer_index}")
        self.layer_path = f"{path}.layers.{layer_index}" if path else f"layers.{layer_index}"
        self.layer_index = int(layer_index)
        self.hidden_states: Tensor | None = None
        self._handle = layers[layer_index].register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise RuntimeError("Early decoder layer did not return [batch, sequence, hidden]")
        self.hidden_states = hidden

    def take_anchors(self, labels: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden = self.hidden_states
        self.hidden_states = None
        if hidden is None:
            raise RuntimeError("Early decoder layer hook did not capture a forward output")
        anchors = prompt_anchor_indices(labels, attention_mask)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, anchors.to(hidden.device)]

    def clear(self) -> None:
        self.hidden_states = None

    def close(self) -> None:
        self.clear()
        self._handle.remove()


def gaussian_soft_targets(
    targets: Tensor,
    *,
    num_classes: int,
    epsilon: float = 0.15,
    tau: float = 1.0,
) -> Tensor:
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be within [0, 1)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    classes = torch.arange(num_classes, device=targets.device, dtype=torch.float32)
    distance = (classes[None, :] - targets[:, None].float()).abs()
    neighbourhood = torch.softmax(-distance / tau, dim=-1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    return (1.0 - epsilon) * one_hot + epsilon * neighbourhood


def inverse_sqrt_class_weights(
    frequencies: Sequence[int], *, minimum: float = 0.5, maximum: float = 2.0
) -> Tensor:
    counts = torch.as_tensor(list(frequencies), dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() == 0 or bool((counts < 0).any()):
        raise ValueError("frequencies must be a non-empty sequence of non-negative counts")
    weights = counts.clamp_min(1).rsqrt()
    weights = weights / weights.mean()
    return weights.clamp(minimum, maximum)


def normalized_cdf_l1(probabilities: Tensor, targets: Tensor) -> Tensor:
    """One-dimensional Wasserstein/CDF distance normalized by count range."""

    target_cdf = (
        torch.nn.functional.one_hot(targets, num_classes=probabilities.shape[-1])
        .float()
        .cumsum(dim=-1)
    )
    predicted_cdf = probabilities.float().cumsum(dim=-1)
    denominator = max(probabilities.shape[-1] - 1, 1)
    return (predicted_cdf - target_cdf).abs().sum(dim=-1) / denominator


def auxiliary_ramp(effective_epoch: float, *, ramp_epochs: float = 0.1) -> float:
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(effective_epoch) / ramp_epochs))


@dataclass(frozen=True)
class CountLossResult:
    total: Tensor
    classification: Tensor
    ordinal: Tensor
    regression: Tensor
    expected_count: Tensor
    target_count: Tensor
    mae: Tensor
    valid_count: int
    auxiliary_weight: float

    def detached_log(self) -> dict[str, float | int]:
        return {
            "loss_count_aux_total": float(self.total.detach().item()),
            "loss_count_classification": float(self.classification.detach().item()),
            "loss_count_ordinal": float(self.ordinal.detach().item()),
            "loss_count_regression": float(self.regression.detach().item()),
            "count_expected_mean": float(self.expected_count.detach().mean().item()),
            "count_target_mean": float(self.target_count.detach().float().mean().item()),
            "count_mae": float(self.mae.detach().item()),
            "count_valid_samples": self.valid_count,
            "count_auxiliary_weight": self.auxiliary_weight,
        }


def categorical_count_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    max_count: int,
    class_weights: Tensor | None = None,
    epsilon: float = 0.15,
    tau: float = 1.0,
    classification_weight: float = 0.5,
    ordinal_weight: float = 1.0,
    regression_weight: float = 0.25,
    auxiliary_weight: float = 1.0,
) -> CountLossResult:
    valid = targets.ne(IGNORE_COUNT_TARGET)
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        empty = targets.new_zeros((1,))
        return CountLossResult(zero, zero, zero, zero, zero[None], empty, zero, 0, auxiliary_weight)
    selected_targets = targets[valid].long().clamp(0, max_count)
    selected_logits = logits[valid]
    if selected_logits.shape[-1] != max_count + 1:
        raise ValueError("Categorical count logits must have K+1 classes")
    soft = gaussian_soft_targets(
        selected_targets,
        num_classes=max_count + 1,
        epsilon=epsilon,
        tau=tau,
    )
    per_sample_classification = -(soft * selected_logits.float().log_softmax(dim=-1)).sum(-1)
    sample_weights = torch.ones_like(per_sample_classification)
    if class_weights is not None:
        sample_weights = class_weights.to(selected_logits.device)[selected_targets]
    classification = (per_sample_classification * sample_weights).mean()
    probabilities = selected_logits.float().softmax(dim=-1)
    ordinal = (normalized_cdf_l1(probabilities, selected_targets) * sample_weights).mean()
    classes = torch.arange(max_count + 1, device=selected_logits.device, dtype=torch.float32)
    expected = (probabilities * classes).sum(-1)
    regression = torch.nn.functional.smooth_l1_loss(
        expected / max(max_count, 1),
        selected_targets.float() / max(max_count, 1),
        beta=1.0,
    )
    mae = (expected - selected_targets.float()).abs().mean()
    combined = auxiliary_weight * (
        classification_weight * classification
        + ordinal_weight * ordinal
        + regression_weight * regression
    )
    return CountLossResult(
        combined,
        classification,
        ordinal,
        regression,
        expected,
        selected_targets,
        mae,
        int(valid.sum().item()),
        auxiliary_weight,
    )


def negative_binomial_nll(parameters: Tensor, targets: Tensor) -> Tensor:
    """Optional NB ablation; never selected unless explicitly configured."""

    if parameters.shape[-1] != 2:
        raise ValueError("Negative-binomial head must emit log-mean and log-dispersion")
    mean = torch.nn.functional.softplus(parameters[:, 0].float()) + 1e-6
    dispersion = torch.nn.functional.softplus(parameters[:, 1].float()) + 1e-6
    values = targets.float()
    log_prob = (
        torch.lgamma(values + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(values + 1.0)
        + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
        + values * (torch.log(mean) - torch.log(dispersion + mean))
    )
    return -log_prob.mean()


def negative_binomial_count_loss(
    parameters: Tensor,
    targets: Tensor,
    *,
    max_count: int,
    class_weights: Tensor | None = None,
    nll_weight: float = 1.0,
    regression_weight: float = 0.25,
    auxiliary_weight: float = 1.0,
) -> CountLossResult:
    """Isolated opt-in NB ablation sharing masking, weighting, ramp, and regression."""

    valid = targets.ne(IGNORE_COUNT_TARGET)
    if not bool(valid.any()):
        zero = parameters.sum() * 0.0
        empty = targets.new_zeros((1,))
        return CountLossResult(zero, zero, zero, zero, zero[None], empty, zero, 0, auxiliary_weight)
    selected_targets = targets[valid].long().clamp(0, max_count)
    selected = parameters[valid]
    mean = torch.nn.functional.softplus(selected[:, 0].float()) + 1e-6
    dispersion = torch.nn.functional.softplus(selected[:, 1].float()) + 1e-6
    values = selected_targets.float()
    per_sample_nll = -(
        torch.lgamma(values + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(values + 1.0)
        + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
        + values * (torch.log(mean) - torch.log(dispersion + mean))
    )
    sample_weights = torch.ones_like(per_sample_nll)
    if class_weights is not None:
        sample_weights = class_weights.to(selected.device)[selected_targets]
    nll = (per_sample_nll * sample_weights).mean()
    regression = torch.nn.functional.smooth_l1_loss(
        mean / max(max_count, 1),
        values / max(max_count, 1),
        beta=1.0,
    )
    mae = (mean - values).abs().mean()
    zero = nll * 0.0
    total = auxiliary_weight * (nll_weight * nll + regression_weight * regression)
    return CountLossResult(
        total,
        nll,
        zero,
        regression,
        mean,
        selected_targets,
        mae,
        int(valid.sum().item()),
        auxiliary_weight,
    )


def parameter_grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return math.sqrt(float(torch.stack(squares).sum().item()))
