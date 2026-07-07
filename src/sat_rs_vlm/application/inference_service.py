"""推理应用服务。

作用：
    承接接口层输入，完成任务路由、模型调用和性能 profile 注入。
    业务逻辑集中在这里，CLI/HTTP 层只负责协议适配。

算法/流程：
    1. 如果 task_type 为 unknown，则调用 TaskRouter 推断任务。
    2. 调用 BaseVLMEngine.infer。
    3. 如果启用 profiler，将性能信息写入 InferenceResult.raw_output["profile"]。
"""

from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.result import InferenceResult
from sat_rs_vlm.domain.task_router import TaskRouter
from sat_rs_vlm.domain.tasks import TaskType
from sat_rs_vlm.infrastructure.config import AppSettings
from sat_rs_vlm.infrastructure.profiler import InferenceProfiler
from sat_rs_vlm.models.base import BaseVLMEngine
from sat_rs_vlm.models.model_factory import create_vlm_engine


class InferenceService:
    """遥感 VLM 推理用例服务。

    参数：
        engine：实现 BaseVLMEngine 协议的模型引擎。
        task_router：可选任务路由器；默认使用关键词路由。
        enable_profiler：是否启用推理性能记录。
        backend：模型后端名称，用于 profiler 标注。
        device：模型设备配置，用于 profiler 标注。
    """

    def __init__(
        self,
        engine: BaseVLMEngine,
        task_router: TaskRouter | None = None,
        *,
        enable_profiler: bool = False,
        backend: str = "mock",
        device: str = "auto",
    ) -> None:
        """初始化应用服务依赖。

        返回值：
            None。
        """

        self._engine = engine
        self._task_router = task_router or TaskRouter()
        self._enable_profiler = enable_profiler
        self._backend = backend
        self._device = device

    @classmethod
    def from_config(cls, config: AppSettings) -> "InferenceService":
        """根据配置创建推理服务。

        参数：
            config：AppSettings，包含模型后端和运行时配置。

        返回值：
            InferenceService：已绑定具体模型引擎的服务实例。
        """

        return cls(
            engine=create_vlm_engine(config.model),
            enable_profiler=config.runtime.enable_profiler,
            backend=config.model.backend,
            device=config.model.device,
        )

    def infer(self, input_data: RemoteSensingInput) -> InferenceResult:
        """执行一次完整推理用例。

        参数：
            input_data：遥感推理输入。

        返回值：
            InferenceResult：统一输出；启用 profiler 时包含 raw_output.profile。
        """

        task_type = input_data.task_type
        if task_type == TaskType.UNKNOWN:
            task_type = self._task_router.route(input_data)
            input_data = input_data.model_copy(update={"task_type": task_type})
        if not self._enable_profiler:
            return self._engine.infer(input_data)

        with InferenceProfiler(backend=self._backend, device=self._device) as profiler:
            result = self._engine.infer(input_data)
        result.raw_output["profile"] = profiler.to_dict()
        return result
