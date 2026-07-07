"""设备解析工具。

作用：
    在基础工程阶段避免引入 torch，仅做轻量字符串解析。真实设备能力探测由
    HuggingFaceVLMEngine 或 InferenceProfiler 在需要时动态完成。
"""


def resolve_device(device: str = "auto") -> str:
    """解析运行设备。

    参数：
        device：用户配置的设备字符串。`auto` 在无重依赖的基础层中解析为 cpu。

    返回值：
        str：解析后的设备名称。
    """

    if device == "auto":
        return "cpu"
    return device
