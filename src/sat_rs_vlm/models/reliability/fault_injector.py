"""state dict 与 safetensors LoRA Adapter 的统一故障注入。

选择器只负责确定允许修改的参数，bit 级修改由唯一的 :mod:`bitflip` 实现完成。
随机地址在全部候选 bit 中无放回抽样，因而固定 seed 可复现，且一次调用不会把同一位
翻转两次。所有 state dict 和 Adapter 输入均保持只读。
"""

from __future__ import annotations

import bisect
import copy
import json
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from sat_rs_vlm.models.reliability.bitflip import flip_tensor_bit, tensor_bit_width
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.models.reliability.schemas import AdapterInjectionReport, BitFlipRecord
from sat_rs_vlm.utils.jsonl import write_jsonl

LoraScope = Literal["none", "all", "a", "b"]
BitPlane = Literal["all", "sign", "exponent", "mantissa"]
FaultTarget = Literal[
    "all_parameters", "lora_adapter", "lora_a", "lora_b", "vision_encoder",
    "language_model", "attention", "mlp", "embeddings",
]


class ParameterChangeSummary(TypedDict):
    """state dict 变化统计的固定字段。"""

    changed_parameters: int
    changed_parameter_names: list[str]
    changed_elements: int
    max_abs_delta: float


@dataclass(frozen=True)
class ParameterSelector:
    """state dict 参数筛选规则；所有非空条件按 AND 组合。

    参数：
        name_contains：名称必须包含其中任一片段。
        name_regex：Python 正则表达式。
        module_names：模型模块名称片段。
        layer_indices：从 `layers.N`、`layer.N`、`blocks.N` 等名称提取的层号。
        parameter_names：精确参数名白名单。
        lora_scope：`all`、`a`、`b` 分别选择全部 LoRA、LoRA A、LoRA B。
    """

    name_contains: tuple[str, ...] = ()
    name_regex: str | None = None
    module_names: tuple[str, ...] = ()
    layer_indices: tuple[int, ...] = ()
    parameter_names: tuple[str, ...] = ()
    lora_scope: LoraScope = "none"

    def matches(self, name: str) -> bool:
        """判断参数名是否同时满足所有已启用条件。"""

        if self.parameter_names and name not in self.parameter_names:
            return False
        if self.name_contains and not any(token in name for token in self.name_contains):
            return False
        if self.name_regex and re.search(self.name_regex, name) is None:
            return False
        if self.module_names and not any(module in name for module in self.module_names):
            return False
        if self.layer_indices:
            matches = re.findall(r"(?:layers?|blocks?)\.(\d+)(?:\.|$)", name)
            if not any(int(index) in self.layer_indices for index in matches):
                return False
        lowered = name.lower()
        if self.lora_scope == "all" and "lora_" not in lowered:
            return False
        if self.lora_scope == "a" and re.search(r"lora_a(?:\.|$)", lowered) is None:
            return False
        if self.lora_scope == "b" and re.search(r"lora_b(?:\.|$)", lowered) is None:
            return False
        return True


def selector_for_fault_target(
    target: FaultTarget | str,
    *,
    name_regex: str | None = None,
    module_names: tuple[str, ...] = (),
    layer_indices: tuple[int, ...] = (),
) -> ParameterSelector:
    """Build a reproducible selector for a named model region."""

    presets: dict[str, dict[str, Any]] = {
        "all_parameters": {},
        "lora_adapter": {"lora_scope": "all"},
        "lora_a": {"lora_scope": "a"},
        "lora_b": {"lora_scope": "b"},
        "vision_encoder": {"module_names": ("visual",)},
        "language_model": {"module_names": ("model.layers",)},
        "attention": {"module_names": ("self_attn",)},
        "mlp": {"module_names": ("mlp",)},
        # Keep visual.patch_embed in the vision target; only language-token
        # embeddings and a language-model lm_head belong to this target.
        "embeddings": {"name_regex": r"(?:language_model\.(?:embed|lm_head)|(?:^|\.)lm_head)"},
    }
    key = str(target).lower()
    if key not in presets:
        raise ValueError("fault.target must be one of: " + ", ".join(sorted(presets)))
    preset = presets[key]
    return ParameterSelector(
        name_regex=name_regex or preset.get("name_regex"),
        module_names=tuple(str(item) for item in preset.get("module_names", ())) + module_names,
        layer_indices=layer_indices,
        lora_scope=preset.get("lora_scope", "none"),
    )


def bit_indices_for_tensor(tensor: Any, bit_plane: BitPlane | str = "all") -> tuple[int, ...]:
    """Return bit positions for a physical floating-point fault category."""

    torch = _torch_module()
    width = tensor_bit_width(tensor)
    plane = str(bit_plane).lower()
    if plane == "all":
        return tuple(range(width))
    layouts: dict[Any, dict[str, tuple[int, ...]]] = {
        torch.float32: {"sign": (31,), "exponent": tuple(range(23, 31)), "mantissa": tuple(range(23))},
        torch.float16: {"sign": (15,), "exponent": tuple(range(10, 15)), "mantissa": tuple(range(10))},
        torch.bfloat16: {"sign": (15,), "exponent": tuple(range(7, 15)), "mantissa": tuple(range(7))},
        torch.int8: {"sign": (7,)},
        torch.uint8: {"sign": (7,)},
    }
    if plane not in {"sign", "exponent", "mantissa"}:
        raise ValueError("fault.bit_plane must be all, sign, exponent or mantissa")
    return layouts.get(tensor.dtype, {}).get(plane, ())



def model_fault_inventory(
    model: Any,
    *,
    selector: ParameterSelector | None = None,
    bit_plane: BitPlane | str = "all",
) -> dict[str, Any]:
    """Describe the exact parameter population eligible for a fault condition."""

    selected = selectable_parameters(dict(model.named_parameters()), selector)
    rows: list[dict[str, Any]] = []
    total_elements = total_bits = 0
    for name, parameter in selected:
        allowed_bits = bit_indices_for_tensor(parameter, bit_plane)
        if not allowed_bits:
            continue
        elements = int(parameter.numel())
        candidate_bits = elements * len(allowed_bits)
        rows.append({
            "name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype).removeprefix("torch."),
            "elements": elements, "bit_positions": list(allowed_bits), "candidate_bits": candidate_bits,
        })
        total_elements += elements
        total_bits += candidate_bits
    return {
        "bit_plane": str(bit_plane), "num_parameters": len(rows),
        "total_elements": total_elements, "candidate_bits": total_bits, "parameters": rows,
    }


def summarize_fault_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate a raw parameter inventory by model region and transformer layer."""

    groups: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in inventory.get("parameters", []):
        name = str(row["name"])
        lowered = name.lower()
        if "lora_" in lowered:
            region = "lora_adapter"
        elif "self_attn" in lowered or "attention" in lowered:
            region = "attention"
        elif ".mlp." in lowered:
            region = "mlp"
        elif "visual" in lowered or "vision" in lowered:
            region = "vision_encoder"
        elif "embed" in lowered or "lm_head" in lowered:
            region = "embeddings"
        elif "model.layers" in lowered:
            region = "language_model"
        else:
            region = "other"
        match = re.search(r"(?:layers?|blocks?)\.(\d+)(?:\.|$)", name)
        layer = int(match.group(1)) if match else None
        item = groups.setdefault(
            (region, layer),
            {"region": region, "layer": layer, "tensor_count": 0, "elements": 0, "candidate_bits": 0},
        )
        item["tensor_count"] += 1
        item["elements"] += int(row["elements"])
        item["candidate_bits"] += int(row["candidate_bits"])
    return sorted(groups.values(), key=lambda item: (str(item["region"]), item["layer"] is None, item["layer"] or -1))



def fault_bits_from_density(candidate_bits: int, flips_per_million_bits: float) -> int:
    """Convert a normalized fault density to an executable count, minimum one."""

    if candidate_bits < 1:
        raise ValueError("candidate_bits must be positive")
    if flips_per_million_bits <= 0:
        raise ValueError("fault density must be positive")
    return max(1, round(candidate_bits * flips_per_million_bits / 1_000_000))



def inject_model_parameter_bitflips(
    model: Any,
    *,
    num_bits: int,
    seed: int,
    selector: ParameterSelector | None = None,
    bit_plane: BitPlane | str = "all",
    bit_index: int | None = None,
    flat_index: int | None = None,
) -> list[BitFlipRecord]:
    """Inject faults into loaded model parameters without cloning full weights.

    This is for one evaluation subprocess: only the selected scalar is copied, then
    process exit discards the in-memory fault. Checkpoint files stay read-only.
    """

    torch = _torch_module()
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    selected = [
        (name, parameter, ((bit_index,) if bit_index is not None else bit_indices_for_tensor(parameter, bit_plane)))
        for name, parameter in selectable_parameters(dict(model.named_parameters()), selector)
    ]
    selected = [item for item in selected if item[2]]
    if not selected:
        raise ValueError("No model parameters matched the fault selector and bit plane")
    if flat_index is not None:
        if len(selected) != 1 or flat_index < 0 or flat_index >= int(selected[0][1].numel()):
            raise ValueError("flat_index requires exactly one matching parameter and a valid index")
    cumulative: list[int] = []
    total = 0
    for _, parameter, allowed_bits in selected:
        total += int(parameter.numel()) * len(allowed_bits)
        cumulative.append(total)
    if flat_index is not None:
        total = len(selected[0][2])
        cumulative = [total]
    if num_bits > total:
        raise ValueError(f"num_bits={num_bits} exceeds candidate bits={total}")

    records: list[BitFlipRecord] = []
    with torch.no_grad():
        for address in random.Random(seed).sample(range(total), num_bits):
            parameter_index = bisect.bisect_right(cumulative, address)
            start = cumulative[parameter_index - 1] if parameter_index else 0
            local_address = address - start
            name, parameter, allowed_bits = selected[parameter_index]
            target_flat_index, bit_rank = divmod(local_address, len(allowed_bits))
            if flat_index is not None:
                target_flat_index = flat_index
            bit_index = allowed_bits[bit_rank]
            width = tensor_bit_width(parameter)
            element = parameter.detach().reshape(-1)[target_flat_index : target_flat_index + 1].clone()
            changed, record = flip_tensor_bit(
                element, flat_index=0, bit_index=bit_index, target_name=name, seed=seed
            )
            parameter.reshape(-1)[target_flat_index].copy_(changed.reshape(-1)[0])
            element_bytes = width // 8
            records.append(record.model_copy(update={
                "flat_index": target_flat_index,
                "byte_index": target_flat_index * element_bytes + bit_index // 8,
                "shape": list(parameter.shape),
            }))
    return records


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("State-dict fault injection requires the 'model' extra") from exc
    return torch


def selectable_parameters(
    state_dict: dict[str, Any],
    selector: ParameterSelector | None = None,
) -> list[tuple[str, Any]]:
    """列出满足规则且 dtype 支持 bit flip 的 tensor 参数。"""

    torch = _torch_module()
    rule = selector or ParameterSelector()
    selected: list[tuple[str, Any]] = []
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not rule.matches(name):
            continue
        tensor_bit_width(value)
        selected.append((name, value))
    return selected


def _clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    torch = _torch_module()
    return {
        name: value.detach().clone().contiguous()
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for name, value in state_dict.items()
    }


def inject_state_dict_bitflips(
    state_dict: dict[str, Any],
    *,
    num_bits: int,
    seed: int,
    selector: ParameterSelector | None = None,
    parameter_name: str | None = None,
    flat_index: int | None = None,
    bit_index: int | None = None,
    bit_plane: BitPlane | str = "all",
) -> tuple[dict[str, Any], list[BitFlipRecord]]:
    """向候选 state dict bit 地址无放回注入故障。

    `parameter_name`、`flat_index`、`bit_index` 可逐级固定目标；未固定的维度由 seed
    决定。返回完整的新 state dict 和记录列表，原字典及其中 tensor 不会被修改。
    """

    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    rule = selector or ParameterSelector()
    if parameter_name is not None:
        if rule.parameter_names and parameter_name not in rule.parameter_names:
            raise ValueError("parameter_name conflicts with selector.parameter_names")
        rule = ParameterSelector(
            name_contains=rule.name_contains,
            name_regex=rule.name_regex,
            module_names=rule.module_names,
            layer_indices=rule.layer_indices,
            parameter_names=(parameter_name,),
            lora_scope=rule.lora_scope,
        )
    selected = selectable_parameters(state_dict, rule)
    if not selected:
        raise ValueError("No tensor parameters matched the fault selector")

    selected = [
        (name, tensor) for name, tensor in selected
        if bit_index is not None or bit_indices_for_tensor(tensor, bit_plane)
    ]
    if not selected:
        raise ValueError("No tensor parameters matched the requested bit plane")
    candidate_sizes: list[int] = []
    for name, tensor in selected:
        width = tensor_bit_width(tensor)
        if flat_index is not None and not 0 <= flat_index < tensor.numel():
            raise ValueError(f"flat_index is outside selected parameter: {name}")
        if bit_index is not None and not 0 <= bit_index < width:
            raise ValueError(f"bit_index is outside dtype width for parameter: {name}")
        allowed_bits = (bit_index,) if bit_index is not None else bit_indices_for_tensor(tensor, bit_plane)
        if not allowed_bits:
            continue
        elements = 1 if flat_index is not None else int(tensor.numel())
        candidate_sizes.append(elements * len(allowed_bits))

    cumulative: list[int] = []
    total = 0
    for size in candidate_sizes:
        total += size
        cumulative.append(total)
    if num_bits > total:
        raise ValueError(f"num_bits={num_bits} exceeds candidate bits={total}")

    updated = _clone_state_dict(state_dict)
    records: list[BitFlipRecord] = []
    for address in random.Random(seed).sample(range(total), num_bits):
        parameter_index = bisect.bisect_right(cumulative, address)
        start = cumulative[parameter_index - 1] if parameter_index else 0
        local_address = address - start
        name, tensor = selected[parameter_index]
        width = tensor_bit_width(tensor)
        allowed_bits = (bit_index,) if bit_index is not None else bit_indices_for_tensor(tensor, bit_plane)
        element_offset, bit_rank = divmod(local_address, len(allowed_bits))
        target_flat_index = flat_index if flat_index is not None else element_offset
        target_bit_index = allowed_bits[bit_rank]
        changed, record = flip_tensor_bit(
            updated[name],
            flat_index=target_flat_index,
            bit_index=target_bit_index,
            target_name=name,
            seed=seed,
        )
        updated[name] = changed
        records.append(record)
    return updated, records


def summarize_parameter_changes(
    clean_state: dict[str, Any],
    changed_state: dict[str, Any],
) -> ParameterChangeSummary:
    """统计发生变化的参数、元素数量和最大绝对差值。"""

    torch = _torch_module()
    changed_names: list[str] = []
    changed_elements = 0
    max_abs_delta = 0.0
    for name, clean in clean_state.items():
        changed = changed_state.get(name)
        if not isinstance(clean, torch.Tensor) or not isinstance(changed, torch.Tensor):
            continue
        if clean.shape != changed.shape:
            raise ValueError(f"Tensor shape changed for parameter: {name}")
        difference_mask = clean.ne(changed)
        count = int(difference_mask.sum().item())
        if count == 0:
            continue
        changed_names.append(name)
        changed_elements += count
        delta = (clean.to(torch.float64) - changed.to(torch.float64)).abs()
        current = float(delta.max().item())
        max_abs_delta = max(max_abs_delta, current)
    return {
        "changed_parameters": len(changed_names),
        "changed_parameter_names": changed_names,
        "changed_elements": changed_elements,
        "max_abs_delta": max_abs_delta,
    }


def load_safetensors_state(path: str | Path) -> tuple[dict[str, Any], dict[str, str] | None]:
    """在 CPU 上加载 safetensors state dict 及文件 metadata。"""

    tensor_path = Path(path)
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Adapter file injection requires safetensors from the 'model' extra"
        ) from exc
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    return dict(load_file(str(tensor_path), device="cpu")), metadata


def save_safetensors_state(
    state_dict: dict[str, Any],
    path: str | Path,
    metadata: dict[str, str] | None,
) -> None:
    """保存连续 tensor，并原样写回可选 safetensors metadata。"""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "Adapter file injection requires safetensors from the 'model' extra"
        ) from exc
    tensors = {name: tensor.contiguous() for name, tensor in state_dict.items()}
    save_file(tensors, str(Path(path)), metadata=metadata)


def _copy_deployable_adapter(source: Path, destination: Path) -> None:
    """Copy only files required to load and audit a deployed LoRA adapter."""

    destination.mkdir(parents=True, exist_ok=False)
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    for entry in source.iterdir():
        if entry.is_file():
            if (
                entry.suffix.lower() in weight_suffixes
                and entry.name != "adapter_model.safetensors"
            ):
                continue
            shutil.copy2(entry, destination / entry.name)
        elif entry.is_dir() and entry.name == "processor":
            shutil.copytree(entry, destination / entry.name)


def inject_safetensors_adapter(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    num_bits: int,
    seed: int,
    selector: ParameterSelector | None = None,
    parameter_name: str | None = None,
    flat_index: int | None = None,
    bit_index: int | None = None,
    bit_plane: BitPlane | str = "all",
    overwrite: bool = False,
) -> AdapterInjectionReport:
    """复制 LoRA Adapter 目录并向副本的 safetensors 权重注入故障。

    原目录始终只读。函数保留 safetensors metadata 和 Adapter 目录中的配置/processor/
    manifest 文件，写出 `fault_records.jsonl` 与 `fault_record.json`，并在发布输出目录前
    完成重载和 hash 验证。
    """

    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Adapter source is not a directory: {source}")
    if source == output or source in output.parents:
        raise ValueError("Fault adapter output must not be the source or a child of the source")
    source_weights = source / "adapter_model.safetensors"
    source_config = source / "adapter_config.json"
    if not source_weights.is_file() or not source_config.is_file():
        raise FileNotFoundError(
            "Adapter source requires adapter_model.safetensors and adapter_config.json"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"Fault adapter output already exists: {output}")
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"Fault adapter output is not a directory: {output}")
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    source_hash_before = file_sha256(source_weights)
    rule = selector or ParameterSelector(lora_scope="all")
    try:
        _copy_deployable_adapter(source, temporary)
        clean_state, metadata = load_safetensors_state(source_weights)
        fault_state, records = inject_state_dict_bitflips(
            clean_state,
            num_bits=num_bits,
            seed=seed,
            selector=rule,
            parameter_name=parameter_name,
            flat_index=flat_index,
            bit_index=bit_index,
            bit_plane=bit_plane,
        )
        fault_weights = temporary / "adapter_model.safetensors"
        save_safetensors_state(fault_state, fault_weights, metadata)
        reloaded, reloaded_metadata = load_safetensors_state(fault_weights)
        if metadata != reloaded_metadata:
            raise RuntimeError("Safetensors metadata changed during fault injection")
        changes = summarize_parameter_changes(clean_state, reloaded)
        changed_names = changes["changed_parameter_names"]
        if not changed_names:
            raise RuntimeError("Fault injection did not change any target parameter")
        source_hash_after = file_sha256(source_weights)
        fault_hash = file_sha256(fault_weights)
        if source_hash_before != source_hash_after:
            raise RuntimeError("Source adapter changed during fault injection")
        if source_hash_before == fault_hash:
            raise RuntimeError("Fault adapter hash unexpectedly equals the source hash")

        report = AdapterInjectionReport(
            source_adapter=str(source),
            fault_adapter=str(output),
            source_sha256_before=source_hash_before,
            source_sha256_after=source_hash_after,
            fault_sha256=fault_hash,
            source_unchanged=True,
            fault_differs=True,
            reload_verified=True,
            changed_parameters=changed_names,
            records=records,
        )
        write_jsonl(
            temporary / "fault_records.jsonl",
            (record.model_dump(mode="json") for record in records),
        )
        (temporary / "fault_record.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.rename(output)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
