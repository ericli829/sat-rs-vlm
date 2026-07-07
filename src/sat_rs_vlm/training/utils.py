"""训练工具函数。

该模块不在导入时加载大模型依赖，只在函数调用时动态导入。
"""

from __future__ import annotations

import importlib
import random
from typing import Any

MODEL_DEPS_ERROR = 'Please install model dependencies with: pip install -e ".[model]"'


def set_seed(seed: int) -> None:
    """设置 Python 和 torch 随机种子。"""

    random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: Any) -> tuple[int, int, float]:
    """统计可训练参数量、总参数量和可训练比例。"""

    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count
    ratio = float(trainable / total) if total else 0.0
    return trainable, total, ratio


def print_trainable_parameters(model: Any) -> None:
    """打印参数统计信息。"""

    trainable, total, ratio = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable}")
    print(f"Total parameters: {total}")
    print(f"Trainable ratio: {ratio:.6f}")


def safe_import_model_dependencies(require_bitsandbytes: bool = False) -> dict[str, Any]:
    """安全导入模型训练依赖。

    缺少依赖时抛出带安装命令的 ImportError。Windows 上 bitsandbytes 不可用时，建议把
    training.method 改为 lora，并设置 qlora.load_in_4bit: false。
    """

    modules = ["torch", "transformers", "peft", "qwen_vl_utils"]
    if require_bitsandbytes:
        modules.append("bitsandbytes")
    imported: dict[str, Any] = {}
    for name in modules:
        try:
            imported[name] = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            hint = MODEL_DEPS_ERROR
            if name == "bitsandbytes":
                hint += " If bitsandbytes is unavailable on Windows, use method='lora'."
            raise ImportError(hint) from exc
    return imported


def torch_device_summary(torch: Any) -> dict[str, Any]:
    """返回 CUDA/显存摘要。"""

    cuda_available = bool(torch.cuda.is_available())
    summary: dict[str, Any] = {"cuda_available": cuda_available, "device_name": None}
    if cuda_available:
        device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_index)
        summary["device_name"] = torch.cuda.get_device_name(device_index)
        summary["total_memory_mb"] = float(props.total_memory / (1024 * 1024))
    return summary
