"""HTTP 路由定义。

作用：
    将 HTTP 请求/响应 schema 转换为领域实体和统一推理结果。该层不引用具体模型类，
    默认使用配置文件中的 backend。
"""

from fastapi import APIRouter

from sat_rs_vlm.application.inference_service import InferenceService
from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.infrastructure.config import load_config
from sat_rs_vlm.interfaces.http.schemas import InferRequest, InferResponse

router = APIRouter()
_service = InferenceService.from_config(load_config())


@router.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。

    返回值：
        dict[str, str]：固定返回 {"status": "ok"}。
    """

    return {"status": "ok"}


@router.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    """HTTP 推理接口。

    参数：
        request：InferRequest，包含 image_path、prompt、second_image_path 等。

    返回值：
        InferResponse：统一推理结果响应模型。
    """

    result = _service.infer(RemoteSensingInput(**request.model_dump()))
    return InferResponse(**result.model_dump())
