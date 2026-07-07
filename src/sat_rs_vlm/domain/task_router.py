"""自然语言任务路由器。

算法：
    使用轻量关键词匹配作为第一阶段/第二阶段的可测试任务识别策略。
    该算法不依赖大模型，适合 CI 和受限设备调试；后续可替换为意图分类模型。
"""

from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.tasks import TaskType


class TaskRouter:
    """基于关键词的遥感任务路由器。

    作用：
        当输入 task_type 为 unknown 时，根据 prompt 粗略判断任务类型。

    算法：
        1. 如果 second_image_path 存在，优先返回 change_detection。
        2. 将 prompt 转为小写。
        3. 按预定义关键词顺序扫描，命中后返回对应 TaskType。
        4. 未命中则返回 unknown。
    """

    _KEYWORDS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
        (TaskType.DETECTION, ("检测", "框出", "位置", "detect", "box", "locate")),
        (TaskType.COUNTING, ("数量", "多少个", "计数", "count", "how many")),
        (TaskType.CHANGE_DETECTION, ("变化", "前后", "新增", "消失", "change", "before", "after")),
        (TaskType.CAPTIONING, ("描述", "说明", "caption", "describe")),
        (TaskType.SCENE_CLASSIFICATION, ("分类", "场景", "classify", "scene")),
        (TaskType.SEGMENTATION, ("分割", "掩膜", "mask", "segment")),
        (TaskType.VQA, ("什么", "是否", "哪里", "why", "what", "where", "is there")),
    )

    def route(self, input_data: RemoteSensingInput) -> TaskType:
        """推断任务类型。

        参数：
            input_data：遥感推理输入，至少包含 prompt 和 image_path。

        返回值：
            TaskType：推断出的任务类型；无法判断时返回 TaskType.UNKNOWN。
        """

        if input_data.second_image_path:
            return TaskType.CHANGE_DETECTION

        prompt = input_data.prompt.lower()
        for task_type, keywords in self._KEYWORDS:
            if any(keyword in prompt for keyword in keywords):
                return task_type
        return TaskType.UNKNOWN
