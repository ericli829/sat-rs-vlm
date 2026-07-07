"""HTTP 请求和响应 schema。

作用：
    使用 Pydantic 定义 API 边界。schema 与领域模型保持相近，但不暴露 backend
    等服务端配置字段，避免客户端控制模型实现细节。
"""

from typing import Any

from pydantic import BaseModel, Field

from sat_rs_vlm.domain.result import BoundingBox
from sat_rs_vlm.domain.tasks import TaskType


class InferRequest(BaseModel):
    """POST /infer 请求体。

    参数：
        image_path：主图像路径。
        prompt：自然语言指令。
        task_type：可选任务类型，默认 unknown，由服务端路由器推断。
        second_image_path：可选第二时相图像路径。
        metadata：扩展元数据。
    """

    image_path: str
    prompt: str
    task_type: TaskType = TaskType.UNKNOWN
    second_image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferResponse(BaseModel):
    """POST /infer 响应体。

    参数：
        task_type：最终任务类型。
        answer：文字回答或摘要。
        boxes：检测框列表。
        masks：分割或变化检测掩膜。
        count：计数结果。
        confidence：整体置信度。
        raw_output：模型后端原始输出、profile 等扩展信息。
    """

    task_type: TaskType
    answer: str | None = None
    boxes: list[BoundingBox] = Field(default_factory=list)
    masks: list[list[list[int]]] = Field(default_factory=list)
    count: int | None = None
    confidence: float | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
