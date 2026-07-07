"""FastAPI 应用工厂。

作用：
    创建 HTTP API 应用并注册路由。业务逻辑位于 routes -> InferenceService，
    该模块只负责应用对象装配。
"""

from fastapi import FastAPI

from sat_rs_vlm.interfaces.http.routes import router


def create_app() -> FastAPI:
    """创建 FastAPI 应用。

    返回值：
        FastAPI：已注册 /health 和 /infer 路由的应用实例。
    """

    app = FastAPI(title="sat-rs-vlm", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
