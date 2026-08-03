"""输出层和权重层保护策略。

`output_guard_vote` 先验证再投票；`clamp_state_dict` 使用干净权重范围生成新的修复副本。
weight clamp 是实验性方法，不等价于航天级纠错编码。
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

from sat_rs_vlm.domain.tasks import TaskType
from sat_rs_vlm.models.reliability.fault_injector import ParameterSelector
from sat_rs_vlm.models.reliability.output_validator import validate_prediction
from sat_rs_vlm.models.reliability.schemas import VoteResult, WeightClampReport


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Weight protection requires the optional 'model' dependencies") from exc
    return torch


def _normalize_vote(task: str, prediction: str, normalized: Any) -> str:
    if task == TaskType.COUNTING.value and isinstance(normalized, dict):
        return f"number:{normalized['count']}"
    if task == TaskType.VQA.value and isinstance(normalized, (int, float)):
        return f"number:{normalized}"
    if isinstance(normalized, str):
        return " ".join(normalized.lower().split())
    return " ".join(prediction.lower().split())


def majority_vote_text(predictions: list[str], *, fallback: str = "") -> VoteResult:
    """对规范化字符串执行多数投票；平票时选择最早出现的候选。"""

    normalized = [" ".join(value.strip().lower().split()) for value in predictions if value.strip()]
    if not normalized:
        return VoteResult(
            selected=fallback,
            has_majority=False,
            used_fallback=True,
            num_inputs=len(predictions),
            num_valid_inputs=0,
        )
    counts = Counter(normalized)
    winning_key = max(counts, key=lambda key: (counts[key], -normalized.index(key)))
    selected = next(
        value for value in predictions if " ".join(value.strip().lower().split()) == winning_key
    )
    return VoteResult(
        selected=selected,
        has_majority=counts[winning_key] > len(normalized) / 2,
        used_fallback=False,
        num_inputs=len(predictions),
        num_valid_inputs=len(normalized),
        votes=dict(counts),
    )


def output_guard_vote(
    task_type: str | TaskType,
    predictions: list[str],
    *,
    fallback: str,
) -> VoteResult:
    """过滤非法输出，再对有效文本、数字或规范化 VQA 回答投票。"""

    task = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    accepted: list[tuple[str, str]] = []
    rejected: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        validation = validate_prediction(task, prediction)
        if validation.valid:
            accepted.append(
                (prediction, _normalize_vote(task, prediction, validation.normalized_output))
            )
        else:
            rejected.append({"index": index, "errors": validation.errors})
    if not accepted:
        return VoteResult(
            selected=fallback,
            has_majority=False,
            used_fallback=True,
            num_inputs=len(predictions),
            num_valid_inputs=0,
            rejected=rejected,
        )
    keys = [key for _, key in accepted]
    counts = Counter(keys)
    winning_key = max(counts, key=lambda key: (counts[key], -keys.index(key)))
    selected = next(original for original, key in accepted if key == winning_key)
    return VoteResult(
        selected=selected,
        has_majority=counts[winning_key] > len(accepted) / 2,
        used_fallback=False,
        num_inputs=len(predictions),
        num_valid_inputs=len(accepted),
        votes=dict(counts),
        rejected=rejected,
    )


def no_protection(clean_prediction: str, fault_prediction: str) -> dict[str, Any]:
    """不修改故障输出，仅记录是否变化，作为保护策略基线。"""

    return {
        "strategy": "no_protection",
        "selected": fault_prediction,
        "changed": clean_prediction.strip() != fault_prediction.strip(),
        "protected": False,
    }


def clamp_state_dict(
    clean_state: dict[str, Any],
    fault_state: dict[str, Any],
    *,
    margin: float = 0.0,
    selector: ParameterSelector | None = None,
) -> tuple[dict[str, Any], WeightClampReport]:
    """按干净参数的 `[min-margin, max+margin]` 生成故障权重修复副本。

    只处理同名、同形状的浮点 tensor。NaN 替换为干净范围中点，正负 Inf 替换为边界。
    clean 和 fault 输入都不会被原地修改。
    """

    if margin < 0:
        raise ValueError("margin must be non-negative")
    torch = _torch_module()
    rule = selector or ParameterSelector()
    protected = {
        name: value.detach().clone().contiguous()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for name, value in fault_state.items()
    }
    processed: list[str] = []
    clipped_elements = 0
    max_adjustment = 0.0
    for name, clean in clean_state.items():
        fault = protected.get(name)
        if (
            not rule.matches(name)
            or not isinstance(clean, torch.Tensor)
            or not isinstance(fault, torch.Tensor)
            or not clean.is_floating_point()
        ):
            continue
        if clean.shape != fault.shape:
            raise ValueError(f"Tensor shape mismatch for weight clamp: {name}")
        lower = float(clean.min().item()) - margin
        upper = float(clean.max().item()) + margin
        midpoint = (lower + upper) / 2
        finite_fault = torch.nan_to_num(fault, nan=midpoint, posinf=upper, neginf=lower)
        clamped = finite_fault.clamp(min=lower, max=upper)
        changed = ~torch.isclose(fault, clamped, rtol=0.0, atol=0.0, equal_nan=True)
        count = int(changed.sum().item())
        processed.append(name)
        clipped_elements += count
        if count:
            finite_delta = (finite_fault - clamped).abs()
            current = float(finite_delta.max().item())
            if math.isfinite(current):
                max_adjustment = max(max_adjustment, current)
        protected[name] = clamped
    return protected, WeightClampReport(
        processed_parameters=processed,
        clipped_elements=clipped_elements,
        max_abs_adjustment=max_adjustment,
    )
