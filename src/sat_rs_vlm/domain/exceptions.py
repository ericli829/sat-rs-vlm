"""项目内异常类型。

作用：
    为配置、推理等错误提供清晰的异常边界，便于 CLI/HTTP 层统一处理。
"""


class SatRsVlmError(Exception):
    """项目基础异常。

    参数：
        message：错误说明，应包含用户下一步可执行的修复动作。
    """


class ConfigurationError(SatRsVlmError):
    """配置加载或校验失败时抛出。"""


class InferenceError(SatRsVlmError):
    """模型推理或结果转换失败时抛出。"""
