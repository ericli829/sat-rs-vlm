"""LoRA 与视觉 merger 的轻量权重诊断工具。

模块只读取 adapter/sidecar 和本地 safetensors，不加载完整 Qwen3-VL 到 GPU。
因此适合在 6 个短实验之间自动执行，分析失败时会返回 ``available=false``，
不会阻断已经完成的训练和评测。
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from sat_rs_vlm.models.reliability.checksum import file_sha256

_LORA_A_RE = re.compile(r"\.lora_A(?:\.default)?\.weight$")
_LORA_B_RE = re.compile(r"\.lora_B(?:\.default)?\.weight$")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _adapter_weight_path(directory: str | Path) -> Path:
    root = Path(directory)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter weight file found under {root}")


def _load_state(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"Unsupported adapter state object: {path}")
    return dict(value)


class _BaseWeightStore:
    """按 safetensors index 延迟读取指定的基座权重。"""

    def __init__(self, model_dir: str | Path) -> None:
        self.root = Path(model_dir)
        index = self.root / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = {}
        if index.is_file():
            payload = json.loads(index.read_text(encoding="utf-8"))
            self.weight_map = dict(payload.get("weight_map", {}))
        self._handles: dict[Path, Any] = {}

    def get(self, key: str) -> Any | None:
        candidates = [key]
        if key.startswith("base_model.model."):
            candidates.append(key.removeprefix("base_model.model."))
        if key.startswith("base_model."):
            candidates.append(key.removeprefix("base_model."))
        if key.startswith("model."):
            candidates.append(key.removeprefix("model."))
        for candidate in candidates:
            shard_name = self.weight_map.get(candidate)
            if shard_name:
                return self._read_from_shard(candidate, self.root / shard_name)
        for shard in sorted(self.root.glob("*.safetensors")):
            try:
                return self._read_from_shard(key, shard)
            except KeyError:
                continue
        return None

    def _read_from_shard(self, key: str, shard: Path) -> Any:
        from safetensors import safe_open

        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        for candidate in (key, key.removeprefix("base_model.model.")):
            if candidate in handle.keys():
                return handle.get_tensor(candidate)
        raise KeyError(key)


def _adapter_prefix(key: str, suffix_re: re.Pattern[str]) -> str | None:
    match = suffix_re.search(key)
    return key[: match.start()] if match else None


def _effective_deltas(directory: str | Path) -> dict[str, Any]:
    """从 LoRA A/B 矩阵恢复每个目标层的有效 delta W。"""

    import torch

    root = Path(directory)
    state = _load_state(_adapter_weight_path(root))
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    rank = float(config.get("r", 1.0) or 1.0)
    scaling = float(config.get("lora_alpha", rank)) / rank
    by_prefix: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, tensor in state.items():
        value = str(key)
        prefix_a = _adapter_prefix(value, _LORA_A_RE)
        prefix_b = _adapter_prefix(value, _LORA_B_RE)
        if prefix_a is not None:
            by_prefix[prefix_a]["a"] = tensor
        elif prefix_b is not None:
            by_prefix[prefix_b]["b"] = tensor
    result: dict[str, Any] = {}
    for prefix, matrices in by_prefix.items():
        if "a" not in matrices or "b" not in matrices:
            continue
        a = matrices["a"].to(dtype=torch.float32)
        b = matrices["b"].to(dtype=torch.float32)
        result[prefix] = b @ a * scaling
    return result


def _svd_summary(matrix: Any) -> dict[str, float | int | None]:
    import torch

    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = float(energy.sum().item())
    if total <= 0.0:
        return {"participation_ratio": 0.0, "top8_energy": 0.0, "r95": 0}
    cumulative = torch.cumsum(energy, dim=0) / total
    r95 = int(torch.searchsorted(cumulative, torch.tensor(0.95)).item() + 1)
    participation = float((singular.sum().square() / energy.sum()).item())
    top8 = float(energy[:8].sum().item() / total)
    return {"participation_ratio": participation, "top8_energy": top8, "r95": r95}


def _norm(value: Any) -> float:
    return float(value.float().norm().item())


def _module_name(prefix: str) -> str:
    return prefix.removeprefix("base_model.model.").removesuffix(".default")


def _pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 2 or len(values_x) != len(values_y):
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in values_x))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in values_y))
    return numerator / (denominator_x * denominator_y) if denominator_x and denominator_y else None


def _rank(values: list[float]) -> list[float]:
    ranked = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranked[ordered[position][0]] = average_rank
        cursor = end
    return ranked


def analyze_lora_adapters(
    initial_adapter: str | Path,
    candidate_adapter: str | Path,
    *,
    model_dir: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """计算 LoRA delta、相对 R1 变化、分层和 projection 统计。"""

    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    initial = _effective_deltas(initial_adapter)
    candidate = _effective_deltas(candidate_adapter)
    base_store = _BaseWeightStore(model_dir) if model_dir is not None else None
    rows: list[dict[str, Any]] = []
    layer_sums: dict[int, dict[str, float]] = defaultdict(lambda: {"delta_sq": 0.0, "base_sq": 0.0})
    projection_sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {"delta_sq": 0.0, "base_sq": 0.0}
    )
    global_delta_sq = global_base_sq = global_change_sq = 0.0
    paired_sq = 0.0
    paired_initial_sq = 0.0
    for prefix, delta in sorted(candidate.items()):
        name = _module_name(prefix)
        initial_delta = initial.get(prefix)
        base_key = prefix.removesuffix(".default") + ".weight"
        base = base_store.get(base_key) if base_store is not None else None
        delta_norm = _norm(delta)
        base_norm = _norm(base) if base is not None else None
        initial_norm = _norm(initial_delta) if initial_delta is not None else None
        change_norm = (
            float((delta.float() - initial_delta.float()).norm().item())
            if initial_delta is not None and tuple(initial_delta.shape) == tuple(delta.shape)
            else None
        )
        global_delta_sq += delta_norm**2
        if base_norm is not None:
            global_base_sq += base_norm**2
        if change_norm is not None:
            global_change_sq += change_norm**2
        if initial_norm is not None:
            paired_sq += float(torch.sum(delta.float() * initial_delta.float()).item())
            paired_initial_sq += initial_norm**2
        layer_match = _LAYER_RE.search(name)
        layer = int(layer_match.group(1)) if layer_match else None
        projection = name.split(".")[-1]
        if layer is not None:
            layer_sums[layer]["delta_sq"] += delta_norm**2
            if base_norm is not None:
                layer_sums[layer]["base_sq"] += base_norm**2
        projection_sums[projection]["delta_sq"] += delta_norm**2
        if base_norm is not None:
            projection_sums[projection]["base_sq"] += base_norm**2
        row: dict[str, Any] = {
            "module": name,
            "layer": layer,
            "projection": projection,
            "delta_frobenius": delta_norm,
            "base_frobenius": base_norm,
            "delta_base_ratio": delta_norm / base_norm if base_norm else None,
            "initial_delta_frobenius": initial_norm,
            "relative_change_from_r1": change_norm / initial_norm if initial_norm else None,
            "r1_cosine": (
                float(
                    torch.nn.functional.cosine_similarity(
                        delta.float().reshape(1, -1), initial_delta.float().reshape(1, -1)
                    ).item()
                )
                if initial_delta is not None and tuple(initial_delta.shape) == tuple(delta.shape)
                else None
            ),
            "svd": _svd_summary(delta),
        }
        rows.append(row)

    global_delta = math.sqrt(global_delta_sq)
    global_base = math.sqrt(global_base_sq)
    global_change = math.sqrt(global_change_sq)
    layer_change_rows = [
        row
        for row in rows
        if isinstance(row.get("layer"), int)
        and isinstance(row.get("relative_change_from_r1"), (int, float))
    ]
    layer_indices = [float(row["layer"]) for row in layer_change_rows]
    layer_changes = [float(row["relative_change_from_r1"]) for row in layer_change_rows]
    summary = {
        "available": bool(rows),
        "initial_adapter": str(initial_adapter),
        "candidate_adapter": str(candidate_adapter),
        "candidate_weight_sha256": file_sha256(_adapter_weight_path(candidate_adapter)),
        "module_count": len(rows),
        "global": {
            "delta_frobenius": global_delta,
            "base_frobenius": global_base if global_base else None,
            "delta_base_ratio": global_delta / global_base if global_base else None,
            "relative_change_from_r1": (
                global_change / math.sqrt(paired_initial_sq) if paired_initial_sq else None
            ),
            "r1_cosine": (
                paired_sq / math.sqrt(global_delta_sq * paired_initial_sq)
                if global_delta_sq and paired_initial_sq
                else None
            ),
        },
        "by_layer": {
            str(layer): {
                "delta_frobenius": math.sqrt(values["delta_sq"]),
                "base_frobenius": math.sqrt(values["base_sq"]) if values["base_sq"] else None,
                "delta_base_ratio": (
                    math.sqrt(values["delta_sq"] / values["base_sq"]) if values["base_sq"] else None
                ),
            }
            for layer, values in sorted(layer_sums.items())
        },
        "by_projection": {
            projection: {
                "delta_frobenius": math.sqrt(values["delta_sq"]),
                "base_frobenius": math.sqrt(values["base_sq"]) if values["base_sq"] else None,
                "delta_base_ratio": (
                    math.sqrt(values["delta_sq"] / values["base_sq"]) if values["base_sq"] else None
                ),
            }
            for projection, values in sorted(projection_sums.items())
        },
        "relative_change_vs_layer": {
            "pearson": _pearson(layer_indices, layer_changes),
            "spearman": _pearson(_rank(layer_indices), _rank(layer_changes)),
            "sample_count": len(layer_change_rows),
        },
        "rows": rows,
    }
    (output / "lora_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(rows, output / "lora_analysis.csv")
    return summary


def analyze_merger_sidecar(
    candidate_checkpoint: str | Path,
    *,
    model_dir: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """分析 main visual merger sidecar 相对基座的绝对/相对更新。"""

    root = Path(candidate_checkpoint)
    sidecar = root / "visual_trainable_weights.safetensors"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not sidecar.is_file():
        result = {"available": False, "reason": "visual sidecar is missing"}
        (output / "merger_analysis.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result
    state = _load_state(sidecar)
    base_store = _BaseWeightStore(model_dir) if model_dir is not None else None
    rows: list[dict[str, Any]] = []
    for key, tuned in sorted(state.items()):
        lowered = key.lower()
        if "merger" not in lowered or "deepstack" in lowered:
            continue
        base_key = key.removeprefix("base_model.model.")
        base = base_store.get(base_key) if base_store is not None else None
        if base is None or tuple(base.shape) != tuple(tuned.shape):
            rows.append({"parameter": key, "available": False})
            continue
        delta = tuned.float() - base.float()
        base_norm = _norm(base)
        delta_norm = _norm(delta)
        rows.append(
            {
                "parameter": key,
                "available": True,
                "absolute_frobenius_delta": delta_norm,
                "base_frobenius": base_norm,
                "relative_frobenius_delta": delta_norm / base_norm if base_norm else None,
            }
        )
    result = {
        "available": bool(rows),
        "candidate_checkpoint": str(candidate_checkpoint),
        "sidecar_sha256": file_sha256(sidecar),
        "main_merger_parameter_count": len(rows),
        "rows": rows,
    }
    (output / "merger_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(rows, output / "merger_analysis.csv")
    return result


def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                    for key, value in row.items()
                }
            )
