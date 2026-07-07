"""统一推理结果 schema。

作用：
    将检测框、分割掩膜、计数、文字回答和原始模型输出收敛到同一个结构，
    使 CLI、HTTP、测试和后续评测模块不需要感知具体模型后端差异。
"""

from typing import Any

from pydantic import BaseModel, Field

from sat_rs_vlm.domain.tasks import TaskType


class BoundingBox(BaseModel):
    """目标检测框。

    参数：
        label：目标类别名称。
        x_min/y_min/x_max/y_max：归一化或像素坐标；当前 Mock 使用归一化坐标。
        confidence：检测置信度，范围通常为 0 到 1。

    返回值：
        可序列化的检测框对象，供 InferenceResult.boxes 聚合。
    """

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0


class InferenceResult(BaseModel):
    """统一推理输出。

    参数：
        task_type：本次推理采用的任务类型。
        answer：生成式回答或任务摘要。
        boxes：目标检测结果列表。
        masks：语义分割/变化检测掩膜；当前用嵌套 int 列表表示轻量占位。
        count：目标计数结果。
        confidence：整体置信度。
        raw_output：后端原始信息、调试信息、性能 profile 等扩展字段。

    返回值：
        Pydantic 模型，可被 CLI/HTTP 直接转成 JSON。
    """

    task_type: TaskType
    answer: str | None = None
    boxes: list[BoundingBox] = Field(default_factory=list)
    masks: list[list[list[int]]] = Field(default_factory=list)
    count: int | None = None
    confidence: float | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
