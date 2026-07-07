"""遥感自然语言任务类型定义。

算法/设计：
    使用 `str, Enum` 而不是普通 Enum，使枚举值可以直接序列化为 JSON 字符串，
    同时兼容 Python 3.10。任务路由、模型输出和 HTTP schema 均复用该类型。

接口：
    TaskType：列举系统支持的遥感解译任务。
"""

from enum import Enum


class TaskType(str, Enum):
    """统一任务枚举。

    作用：
        约束输入实体和推理结果中的 task_type 字段，避免层间传递裸字符串。

    返回值/序列化：
        Pydantic/JSON 序列化时输出枚举的字符串值，例如 `"detection"`。
    """

    DETECTION = "detection"
    SCENE_CLASSIFICATION = "scene_classification"
    SEGMENTATION = "segmentation"
    CHANGE_DETECTION = "change_detection"
    COUNTING = "counting"
    CAPTIONING = "captioning"
    VQA = "vqa"
    UNKNOWN = "unknown"
