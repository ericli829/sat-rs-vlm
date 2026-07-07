"""模型引擎抽象接口。

作用：
    定义应用层依赖的最小模型协议。业务服务只依赖 BaseVLMEngine，不关心
    具体实现是 Mock、HuggingFace 还是未来的量化/蒸馏模型。
"""

from typing import Protocol

from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.result import InferenceResult


class BaseVLMEngine(Protocol):
    """遥感多模态大模型推理协议。

    实现要求：
        所有模型后端必须实现 infer，并返回统一 InferenceResult。
    """

    def infer(self, input_data: RemoteSensingInput) -> InferenceResult:
        """执行遥感多模态推理。

        参数：
            input_data：RemoteSensingInput，包含图像路径、prompt、任务类型等。

        返回值：
            InferenceResult：统一推理结果。
        """
