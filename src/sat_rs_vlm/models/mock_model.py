"""Mock 多模态遥感模型。

算法：
    根据 TaskType 返回确定性的模拟结果，不读取真实图像、不依赖 GPU。
    该实现用于本地开发、CI、接口联调和上层业务逻辑验证。
"""

from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.result import BoundingBox, InferenceResult
from sat_rs_vlm.domain.tasks import TaskType
from sat_rs_vlm.models.base import BaseVLMEngine


class MockVLMEngine(BaseVLMEngine):
    """确定性 Mock VLM 引擎。

    作用：
        在未安装真实模型依赖时提供始终可用的推理后端。
    """

    def infer(self, input_data: RemoteSensingInput) -> InferenceResult:
        """根据任务类型返回模拟推理结果。

        参数：
            input_data：遥感推理输入；task_type 应由 InferenceService/TaskRouter 设置。

        返回值：
            InferenceResult：包含 answer、boxes、masks、count 等字段的统一结果。
        """

        task_type = input_data.task_type
        raw = {
            "engine": "mock",
            "image_path": input_data.image_path,
            "second_image_path": input_data.second_image_path,
            "prompt": input_data.prompt,
        }

        if task_type == TaskType.DETECTION:
            return InferenceResult(
                task_type=task_type,
                answer="检测到疑似建筑物、道路和开阔地目标。",
                boxes=[
                    BoundingBox(
                        label="building",
                        x_min=0.12,
                        y_min=0.18,
                        x_max=0.34,
                        y_max=0.42,
                        confidence=0.86,
                    ),
                    BoundingBox(
                        label="road",
                        x_min=0.50,
                        y_min=0.10,
                        x_max=0.82,
                        y_max=0.22,
                        confidence=0.79,
                    ),
                ],
                confidence=0.82,
                raw_output=raw,
            )
        if task_type == TaskType.COUNTING:
            return InferenceResult(
                task_type=task_type,
                answer="估计目标数量为 5。",
                count=5,
                confidence=0.81,
                raw_output=raw,
            )
        if task_type == TaskType.CHANGE_DETECTION:
            return InferenceResult(
                task_type=task_type,
                answer="检测到新增建筑区域和局部道路变化。",
                masks=[[[0, 0, 1], [0, 1, 1], [0, 0, 0]]],
                confidence=0.78,
                raw_output=raw,
            )
        if task_type == TaskType.SEGMENTATION:
            return InferenceResult(
                task_type=task_type,
                answer="已生成建筑、道路、水体的模拟分割结果。",
                masks=[[[1, 1, 0], [0, 2, 2], [3, 3, 0]]],
                confidence=0.80,
                raw_output=raw,
            )
        if task_type == TaskType.SCENE_CLASSIFICATION:
            return InferenceResult(
                task_type=task_type,
                answer="场景类别为城市建设区。",
                confidence=0.88,
                raw_output={**raw, "label": "urban"},
            )
        if task_type == TaskType.VQA:
            return InferenceResult(
                task_type=task_type,
                answer="根据模拟推理，图像中存在道路和建筑物。",
                confidence=0.74,
                raw_output=raw,
            )
        if task_type == TaskType.CAPTIONING:
            return InferenceResult(
                task_type=task_type,
                answer="这是一幅包含建筑群、道路和少量植被的遥感图像。",
                confidence=0.84,
                raw_output=raw,
            )

        return InferenceResult(
            task_type=TaskType.UNKNOWN,
            answer="无法确定任务类型，已返回通用遥感图像解译结果。",
            confidence=0.50,
            raw_output=raw,
        )
