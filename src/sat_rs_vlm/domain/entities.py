"""领域输入实体定义。

作用：
    描述一次遥感多模态推理请求的核心业务数据，不绑定 CLI、HTTP 或具体模型。

接口：
    RemoteSensingInput：应用层和模型层之间传递的统一输入对象。
"""

from typing import Any

from pydantic import BaseModel, Field

from sat_rs_vlm.domain.tasks import TaskType


class RemoteSensingInput(BaseModel):
    """遥感推理输入。

    参数：
        image_path：主图像路径。第一阶段不强制校验文件存在，便于 Mock 和 CI。
        prompt：自然语言指令，例如“请检测图像中的飞机”。
        task_type：任务类型；默认 unknown，交由 TaskRouter 根据 prompt 推断。
        second_image_path：第二时相图像路径；存在时优先视为变化检测任务。
        metadata：扩展元数据，例如传感器、分辨率、轨道号、区域 ID 等。

    返回值：
        Pydantic 模型实例，可通过 model_dump(mode="json") 转为 JSON 兼容字典。
    """

    image_path: str
    prompt: str
    task_type: TaskType = TaskType.UNKNOWN
    second_image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
