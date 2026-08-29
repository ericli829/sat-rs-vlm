"""Minimal formal CLI for single-sample TaskGraph Runtime execution."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml

from .runtime import RuntimeRequest, runtime_from_config
from .runtime_types import ChoiceResult, runtime_summary
from .schema import QuestionType, parse_taskgraph

app = typer.Typer(help="High-resolution TaskGraph Runtime")


def _load_json_value(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.is_file() else value
    return json.loads(text)


@app.command("run")
def run(
    image: list[str] = typer.Option(  # noqa: B008
        ..., "--image", help="Image path; repeat for multi-image."
    ),
    question: str = typer.Option(..., "--question"),
    dataset: str = typer.Option("MME_RealWorld_RS", "--dataset"),
    task: str = typer.Option("default", "--task"),
    sample_id: str = typer.Option("sample", "--sample-id"),
    options_json: str | None = typer.Option(None, "--options-json"),
    graph_json: str | None = typer.Option(None, "--graph-json"),
    provider_config: str = typer.Option("configs/taskgraph/runtime.fake.yaml", "--provider-config"),
    target: str | None = typer.Option(None, "--target"),
    question_type: QuestionType = typer.Option(  # noqa: B008
        QuestionType.FREE_FORM, "--question-type"
    ),
    trace_output: str | None = typer.Option(None, "--trace-output"),
    real_model: bool = typer.Option(
        False, "--real-model", help="Acknowledge real provider config."
    ),
) -> None:
    """Execute one dataset sample through DIRECT or TASKGRAPH_UHR routing."""

    config_path = Path(provider_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    providers = config.get("providers", {})
    configured_real = any(
        isinstance(section, dict) and section.get("kind") not in {None, "fake", "fixture"}
        for section in providers.values()
    )
    if configured_real and not real_model:
        raise typer.BadParameter("real providers require explicit --real-model")
    options = tuple(str(item) for item in _load_json_value(options_json, []))
    graph = parse_taskgraph(_load_json_value(graph_json, {})) if graph_json else None
    runtime = runtime_from_config(config)
    try:
        result = runtime.run(
            RuntimeRequest(
                sample_id=sample_id,
                dataset=dataset,
                task_category=task,
                question=question,
                image_paths=tuple(image),
                options=options,
                question_type=question_type,
                target_category=target,
                graph=graph,
            )
        )
        if trace_output:
            result.trace.write_json(trace_output)
        if isinstance(result.output, ChoiceResult):
            output = asdict(result.output)
        elif isinstance(result.output, tuple):
            output = [runtime_summary(item) for item in result.output]
        else:
            output = runtime_summary(result.output)
        typer.echo(
            json.dumps(
                {
                    "execution_mode": result.execution_mode.value,
                    "output": output,
                    "trace": result.trace.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    app()
