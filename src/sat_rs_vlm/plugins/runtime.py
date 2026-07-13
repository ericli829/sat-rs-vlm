"""仅供显式外部插件命令调用的训练生命周期。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.plugins.api import ExternalFineTuningPlugin
from sat_rs_vlm.plugins.capability import validate_platform_capability
from sat_rs_vlm.plugins.context import PluginContext
from sat_rs_vlm.plugins.discovery import DiscoveredPlugin
from sat_rs_vlm.plugins.errors import PluginExecutionError
from sat_rs_vlm.plugins.services import build_public_services


def run_external_plugin_from_local_directory(plugin_dir: Path, command: str) -> int:
    """供插件内薄脚本使用，不复制发现、依赖或训练流程。"""

    directory = plugin_dir.resolve()
    project_root = Path(__file__).resolve().parents[3]
    if command == "train":
        target = project_root / "scripts" / "run_external_strategy.py"
        arguments = [
            sys.executable,
            str(target),
            "--plugin-root",
            str(directory.parents[1]),
            "--strategy",
            directory.name,
            *sys.argv[1:],
        ]
    elif command == "evaluate":
        target = project_root / "scripts" / "evaluate_rs_vlm.py"
        arguments = [sys.executable, str(target), *sys.argv[1:]]
    else:
        raise ValueError(f"Unsupported local plugin command: {command}")
    return subprocess.run(arguments, cwd=project_root, check=False).returncode


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable: {expanded}")
        return expanded
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_external_config(path: Path) -> dict[str, Any]:
    """读取插件训练 YAML 并展开环境变量。"""

    if not path.is_file():
        raise FileNotFoundError(f"Plugin training config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(_expand_env(payload))


def apply_external_overrides(
    config: dict[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """统一 CLI 字段覆盖插件 YAML。"""

    result = deepcopy(config)
    mapping = {
        "model_dir": ("model", "model_dir"),
        "processor_dir": ("model", "processor_dir"),
        "train_file": ("data", "train_file"),
        "val_file": ("data", "val_file"),
        "image_root": ("data", "image_root"),
        "max_train_samples": ("data", "max_train_samples"),
        "max_eval_samples": ("data", "max_eval_samples"),
        "max_steps": ("training", "max_steps"),
        "resume_from_checkpoint": ("training", "resume_from_checkpoint"),
    }
    for cli_name, (section, field) in mapping.items():
        value = overrides.get(cli_name)
        if value is not None:
            result.setdefault(section, {})[field] = value
    if bool(overrides.get("skip_eval", False)):
        result.setdefault("evaluation", {})["do_eval"] = False
    return result


def _safe_output_dir(
    plugin: DiscoveredPlugin,
    config: Mapping[str, Any],
    explicit_output: str | None,
) -> Path:
    if explicit_output:
        path = Path(explicit_output).expanduser()
        resolved = (path if path.is_absolute() else plugin.directory / path).resolve()
    else:
        experiment = str(config.get("experiment", {}).get("name", "run"))
        resolved = (plugin.directory / plugin.manifest.paths.checkpoints_dir / experiment).resolve()
    safe_root = plugin.directory.resolve()
    if resolved != safe_root and safe_root not in resolved.parents:
        raise PluginExecutionError(
            plugin_name=plugin.manifest.plugin.name,
            stage="output_path",
            reason=f"output directory is outside the current plugin directory: {resolved}",
            suggested_action="Choose an output path inside this plugin's own directory.",
        )
    return resolved


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def build_plugin_context(
    *,
    plugin: DiscoveredPlugin,
    config: Mapping[str, Any],
    output_dir: Path,
    project_root: Path,
    dry_run: bool,
    forward_only: bool,
    install_missing: bool,
    require_bitsandbytes: bool,
) -> PluginContext:
    """构造只有白名单服务的只读上下文。"""

    model = dict(config.get("model", {}))
    data = dict(config.get("data", {}))
    model_dir = _path(model["model_dir"], project_root)
    processor_dir = _path(model.get("processor_dir", model["model_dir"]), project_root)
    val_value = data.get("val_file")
    return PluginContext(
        project_root=project_root,
        plugin_root=plugin.root,
        plugin_dir=plugin.directory,
        output_dir=output_dir,
        model_dir=model_dir,
        processor_dir=processor_dir,
        train_file=_path(data["train_file"], project_root),
        val_file=_path(val_value, project_root) if val_value else None,
        image_root=_path(data["image_root"], project_root),
        device="auto",
        dry_run=dry_run,
        forward_only=forward_only,
        install_missing=install_missing,
        common_services=build_public_services(require_bitsandbytes=require_bitsandbytes),
    )


def _validate_assets(context: PluginContext) -> None:
    for label, path in (
        ("model_dir", context.model_dir),
        ("processor_dir", context.processor_dir),
        ("image_root", context.image_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not context.train_file.is_file():
        raise FileNotFoundError(f"train_file does not exist: {context.train_file}")
    if context.val_file is not None and not context.val_file.is_file():
        raise FileNotFoundError(f"val_file does not exist: {context.val_file}")


def _platform_check(plugin: DiscoveredPlugin, context: PluginContext) -> None:
    validate_platform_capability(plugin.manifest)


def _write_run_metadata(
    plugin: DiscoveredPlugin,
    context: PluginContext,
    config: Mapping[str, Any],
) -> None:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    (context.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copyfile(plugin.directory / "plugin.yaml", context.output_dir / "plugin.yaml")
    if context.dry_run:
        environment = {"runtime_probe": "skipped for dry_run"}
    else:
        try:
            environment = context.service("inspect_environment")()
        except (ImportError, OSError, RuntimeError) as exc:
            environment = {"runtime_probe_error": f"{type(exc).__name__}: {exc}"}
    context.service("write_json_report")(
        context.output_dir / "environment.json",
        {"python": sys.version, "executable": sys.executable, **environment},
    )


def execute_external_plugin(
    *,
    discovered: DiscoveredPlugin,
    plugin: ExternalFineTuningPlugin,
    config: dict[str, Any],
    project_root: Path,
    output_dir: str | None,
    dry_run: bool,
    forward_only: bool,
    skip_eval: bool,
    install_missing: bool,
    require_bitsandbytes: bool,
) -> dict[str, Any]:
    """运行外部策略；默认 LoRA 从不调用此函数。"""

    resolved_output = _safe_output_dir(discovered, config, output_dir)
    context = build_plugin_context(
        plugin=discovered,
        config=config,
        output_dir=resolved_output,
        project_root=project_root,
        dry_run=dry_run,
        forward_only=forward_only,
        install_missing=install_missing,
        require_bitsandbytes=require_bitsandbytes,
    )
    started = time.perf_counter()
    _validate_assets(context)
    _platform_check(discovered, context)
    plugin.validate(context, config)
    _write_run_metadata(discovered, context, config)
    if dry_run:
        report = {"success": True, "status": "passed", "mode": "dry_run"}
        context.service("write_json_report")(context.output_dir / "train_report.json", report)
        return report
    processor = context.service("load_processor")(context, config)
    model_kwargs = plugin.model_load_kwargs(context, config)
    model = context.service("load_base_model")(context, config, model_kwargs)
    model = plugin.prepare_model(context, model, processor, config)
    data = dict(config.get("data", {}))
    train_dataset = context.service("create_dataset")(
        context.train_file,
        data.get("max_train_samples"),
    )
    eval_dataset = None
    if not skip_eval and context.val_file is not None:
        eval_dataset = context.service("create_dataset")(
            context.val_file,
            data.get("max_eval_samples"),
        )
    collator = context.service("create_collator")(
        processor,
        int(data.get("max_seq_length", 1024)),
        context.image_root,
    )
    forward_loss = context.service("forward_probe")(model, collator, train_dataset[0])
    summary = context.service("parameter_summary")(model)
    strategy_manifest = {
        "schema_version": "1",
        "external_plugin": True,
        "strategy": plugin.name,
        "plugin_version": plugin.version,
        "api_version": plugin.api_version,
        "adapter_based": discovered.manifest.capabilities.adapter_based,
        "quantized_base": discovered.manifest.capabilities.quantized_base,
        "supports_merge": discovered.manifest.capabilities.supports_merge,
        "model_dir": str(context.model_dir),
        "processor_dir": str(context.processor_dir),
        **summary,
        **dict(plugin.report_details(context, model, config)),
    }
    context.service("write_json_report")(
        context.output_dir / discovered.manifest.outputs.manifest_file,
        strategy_manifest,
    )
    if forward_only:
        report = {
            "success": True,
            "status": "passed",
            "mode": "forward_only",
            "forward_loss": forward_loss,
            **summary,
        }
        context.service("write_json_report")(context.output_dir / "train_report.json", report)
        return report
    arguments = plugin.build_training_arguments(context, config)
    optimizer_groups = plugin.optimizer_parameter_groups(context, model, config)
    trainer = context.service("create_trainer")(
        model=model,
        context=context,
        arguments=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        optimizer_groups=optimizer_groups,
        callbacks=plugin.trainer_callbacks(context, config),
    )
    resume = config.get("training", {}).get("resume_from_checkpoint")
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = dict(getattr(result, "metrics", {}) or {})
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())
    plugin.save_artifacts(context, model, processor, context.output_dir)
    trainer.save_state()
    plugin_metrics = plugin.evaluate(context, context.output_dir, config)
    if plugin_metrics:
        metrics.update(plugin_metrics)
    report = {
        "success": True,
        "status": "passed",
        "strategy": plugin.name,
        "forward_loss": forward_loss,
        "duration_seconds": time.perf_counter() - started,
        "metrics": metrics,
        **summary,
        **dict(plugin.report_details(context, model, config)),
    }
    context.service("write_json_report")(context.output_dir / "metrics.json", metrics)
    context.service("write_json_report")(context.output_dir / "train_report.json", report)
    return report
