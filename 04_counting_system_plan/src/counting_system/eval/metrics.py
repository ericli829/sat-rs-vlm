"""计数评测指标：exact accuracy、MAE，以及检测诊断。"""

from __future__ import annotations

import subprocess
from typing import Any, Iterable, Sequence


def summarize_counts(
    pairs: Sequence[tuple[int | None, int | None]],
) -> dict[str, Any]:
    """pairs: (pred, ref)。解析失败的 pred 记 exact 错，不进入 MAE。"""
    n = len(pairs)
    parsed = [(p, r) for p, r in pairs if p is not None and r is not None]
    exact = sum(1 for p, r in pairs if p is not None and r is not None and p == r)
    within1 = sum(1 for p, r in parsed if abs(p - r) <= 1)
    abs_err = [abs(p - r) for p, r in parsed]
    signed = [p - r for p, r in parsed]
    mae = sum(abs_err) / len(abs_err) if abs_err else None
    rmse = (sum(e * e for e in abs_err) / len(abs_err)) ** 0.5 if abs_err else None
    return {
        "num_samples": n,
        "parsed": len(parsed),
        "parse_rate": (len(parsed) / n) if n else 0.0,
        "exact_accuracy": (exact / n) if n else 0.0,
        "within1_accuracy": (within1 / n) if n else 0.0,
        "mae": mae,
        "rmse": rmse,
        "mean_signed_error": (sum(signed) / len(signed)) if signed else None,
    }


def choice_match(pred_count: int, options: Sequence[str], answer_letter: str) -> bool:
    if not answer_letter or not options:
        return False
    idx = ord(answer_letter.upper()) - ord("A")
    if not (0 <= idx < len(options)):
        return False
    text = options[idx]
    return str(pred_count) in text or text.strip() == str(pred_count)


def detection_prf(
    pred_boxes: Iterable[tuple[float, float, float, float]],
    gt_boxes: Iterable[tuple[float, float, float, float]],
    iou_thr: float = 0.5,
) -> dict[str, float]:
    from ..geometry import iou

    preds = list(pred_boxes)
    gts = list(gt_boxes)
    if not preds and not gts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    matched_gt: set[int] = set()
    tp = 0
    for pb in preds:
        best_i = -1
        best = 0.0
        for i, gb in enumerate(gts):
            if i in matched_gt:
                continue
            score = iou(pb, gb)
            if score > best:
                best, best_i = score, i
        if best >= iou_thr and best_i >= 0:
            tp += 1
            matched_gt.add(best_i)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def gpu_mem_snapshot() -> dict[str, Any]:
    """当前进程 CUDA 峰值 + nvidia-smi 瞬时占用。峰值以 torch max 为准。"""
    stats: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            stats["device"] = torch.cuda.get_device_name(0)
            stats["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            stats["torch_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            stats["torch_max_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        used, total, util = [part.strip() for part in out.split(",")[:3]]
        stats["nvidia_smi_used_mb"] = float(used)
        stats["nvidia_smi_total_mb"] = float(total)
        stats["nvidia_smi_util"] = float(util)
    except Exception:
        pass
    return stats


def merge_gpu_peak(peak: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
    out = dict(peak)
    for key, value in snap.items():
        if isinstance(value, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = max(out[key], value)
        elif key not in out:
            out[key] = value
    return out
