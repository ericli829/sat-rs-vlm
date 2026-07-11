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


def model_input_device(model: Any, torch: Any) -> Any:
    """返回模型输入 token 应放置的设备。

    算法：
        优先读取输入嵌入层权重设备，因为 `device_map="auto"` 可能把不同模块分配到
        不同设备；若模型不暴露输入嵌入层，则回退到首个参数设备，最后根据 CUDA
        可用性选择 `cuda:0` 或 `cpu`。

    参数：
        model：已经加载的 HuggingFace/PyTorch 模型。
        torch：动态导入的 torch 模块。

    返回值：
        torch.device：输入张量应移动到的设备。
    """

    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None and str(device) != "meta":
            return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        first_parameter = next(iter(parameters()), None)
        device = getattr(first_parameter, "device", None)
        if device is not None and str(device) != "meta":
            return device

    fallback = "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
    return torch.device(fallback)


def move_to_device(value: Any, device: Any, torch: Any) -> Any:
    """递归地把 batch 中的 tensor 移动到指定设备。

    参数：
        value：tensor、字典、列表、元组或其他保持不变的值。
        device：目标 torch.device。
        torch：动态导入的 torch 模块，用于判断 tensor。

    返回值：
        Any：结构保持不变、其中 tensor 已移动到目标设备的对象。
    """

    if bool(torch.is_tensor(value)):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, torch) for item in value)
    return value


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
