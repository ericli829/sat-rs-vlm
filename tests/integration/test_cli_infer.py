import json

from typer.testing import CliRunner

from sat_rs_vlm.interfaces.cli import app


def test_cli_infer_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "infer",
            "--image",
            "examples/demo_image.jpg",
            "--prompt",
            "请描述这张遥感图像中的主要地物。",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["task_type"] == "captioning"
    assert payload["answer"]
    assert "profile" in payload["raw_output"]


def test_cli_infer_backend_override_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "infer",
            "--backend",
            "mock",
            "--image",
            "examples/demo_image.jpg",
            "--prompt",
            "请检测图像中的飞机。",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["task_type"] == "detection"
    assert payload["answer"]
