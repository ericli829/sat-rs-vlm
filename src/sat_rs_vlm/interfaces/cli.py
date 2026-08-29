"""命令行接口。

作用：
    使用 Typer 将本地命令转换为应用层调用。该模块不直接实例化具体模型类，
    只加载配置并通过 InferenceService.from_config 创建服务。
"""

import json
from typing import Any, cast

import typer

from sat_rs_vlm.application.inference_service import InferenceService
from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.infrastructure.config import load_config
from sat_rs_vlm.taskgraph.cli import app as taskgraph_app

app = typer.Typer(help="sat-rs-vlm command line interface")
app.add_typer(taskgraph_app, name="taskgraph")


def _jsonable(value: Any) -> dict[str, Any]:
    """将 Pydantic 对象转换为 JSON 兼容字典。

    参数：
        value：支持 model_dump(mode="json") 的 Pydantic 对象。

    返回值：
        dict[str, Any]：可传给 json.dumps 的字典。
    """

    return cast(dict[str, Any], value.model_dump(mode="json"))


@app.command("config")
def show_config() -> None:
    """打印当前配置。

    返回值：
        None。结果通过 stdout 输出 JSON。
    """

    settings = load_config()
    typer.echo(json.dumps(_jsonable(settings), ensure_ascii=False, indent=2))


@app.command("infer")
def infer(
    image: str = typer.Option(..., "--image", help="Path to the primary remote-sensing image."),
    prompt: str = typer.Option(..., "--prompt", help="Natural-language instruction."),
    config: str = typer.Option("configs/default.yaml", "--config", help="Path to YAML config."),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Override model backend: mock or huggingface.",
    ),
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="Override HuggingFace model id.",
    ),
    second_image: str | None = typer.Option(
        None,
        "--second-image",
        help="Optional second image for change detection.",
    ),
) -> None:
    """执行 CLI 推理。

    参数：
        image：主图像路径。
        prompt：自然语言指令。
        config：YAML 配置路径。
        backend：可选后端覆盖值，支持 mock 或 huggingface。
        model_id：可选模型 ID 覆盖值，主要用于 huggingface 后端。
        second_image：可选第二时相图像路径。

    返回值：
        None。推理结果通过 stdout 输出 JSON。
    """

    settings = load_config(config)
    if backend is not None:
        normalized_backend = backend.lower()
        if normalized_backend not in {"mock", "huggingface"}:
            raise typer.BadParameter("--backend must be either 'mock' or 'huggingface'.")
        settings = settings.model_copy(
            update={"model": settings.model.model_copy(update={"backend": normalized_backend})}
        )
    if model_id is not None:
        settings = settings.model_copy(
            update={"model": settings.model.model_copy(update={"model_id": model_id})}
        )
    service = InferenceService.from_config(settings)
    result = service.infer(
        RemoteSensingInput(image_path=image, prompt=prompt, second_image_path=second_image)
    )
    typer.echo(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
