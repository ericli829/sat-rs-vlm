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

    candidate_sizes: list[int] = []
    for name, tensor in selected:
        width = tensor_bit_width(tensor)
        if flat_index is not None and not 0 <= flat_index < tensor.numel():
            raise ValueError(f"flat_index is outside selected parameter: {name}")
        if bit_index is not None and not 0 <= bit_index < width:
            raise ValueError(f"bit_index is outside dtype width for parameter: {name}")
        elements = 1 if flat_index is not None else int(tensor.numel())
        bits = 1 if bit_index is not None else width
        candidate_sizes.append(elements * bits)

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
        bits_per_candidate = 1 if bit_index is not None else width
        element_offset, local_bit = divmod(local_address, bits_per_candidate)
        target_flat_index = flat_index if flat_index is not None else element_offset
        target_bit_index = bit_index if bit_index is not None else local_bit
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
