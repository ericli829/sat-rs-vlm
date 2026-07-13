"""外部插件 YAML manifest 模型与路径安全验证。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from sat_rs_vlm.plugins.api import EXTERNAL_PLUGIN_API_VERSION
from sat_rs_vlm.plugins.errors import PluginCompatibilityError, PluginValidationError

PLUGIN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class PluginMetadata(BaseModel):
    name: str
    display_name: str
    version: str
    description: str
    api_version: str
    status: str = "experimental"

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not PLUGIN_NAME_PATTERN.fullmatch(value):
            raise ValueError("must contain only lowercase letters, digits, '_' or '-'")
        return value


class EntrypointConfig(BaseModel):
    module: str
    class_name: str = Field(alias="class")


class CompatibilityConfig(BaseModel):
    python: str = ">=3.10"
    platforms: list[str] = Field(default_factory=lambda: ["linux", "windows", "darwin"])
    requires_cuda: bool = False
    supports_cpu: bool = True
    supports_local_model: bool = True
    supports_offline_mode: bool = True


class DependencyConfig(BaseModel):
    requirements_file: str = "requirements.txt"
    optional_requirements_files: list[str] = Field(default_factory=list)
    install_mode: str = "current_venv"


class PluginPaths(BaseModel):
    default_train_config: str = "configs/train.yaml"
    default_smoke_config: str = "configs/smoke.yaml"
    checkpoints_dir: str = "checkpoints"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    docs_dir: str = "docs"


class PluginCapabilities(BaseModel):
    adapter_based: bool
    quantized_base: bool = False
    supports_resume: bool = True
    supports_forward_only: bool = True
    supports_smoke_train: bool = True
    supports_evaluation: bool = True
    supports_merge: bool = False


class PluginOutputs(BaseModel):
    manifest_file: str = "strategy_manifest.json"
    train_report_file: str = "train_report.json"
    evaluation_report_file: str = "evaluation_report.json"


class ExternalPluginManifest(BaseModel):
    schema_version: str
    plugin: PluginMetadata
    entrypoint: EntrypointConfig
    compatibility: CompatibilityConfig = Field(default_factory=CompatibilityConfig)
    dependencies: DependencyConfig = Field(default_factory=DependencyConfig)
    paths: PluginPaths
    capabilities: PluginCapabilities
    outputs: PluginOutputs = Field(default_factory=PluginOutputs)

    @model_validator(mode="after")
    def supported_versions(self) -> ExternalPluginManifest:
        if self.schema_version != "1":
            raise ValueError(f"unsupported schema_version {self.schema_version!r}; expected '1'")
        if self.plugin.api_version != EXTERNAL_PLUGIN_API_VERSION:
            raise ValueError(
                f"unsupported api_version {self.plugin.api_version!r}; "
                f"expected {EXTERNAL_PLUGIN_API_VERSION!r}"
            )
        return self

    def relative_paths(self) -> list[str]:
        """返回清单中必须限制在插件目录内的全部路径。"""

        return [
            self.entrypoint.module,
            self.dependencies.requirements_file,
            *self.dependencies.optional_requirements_files,
            self.paths.default_train_config,
            self.paths.default_smoke_config,
            self.paths.checkpoints_dir,
            self.paths.reports_dir,
            self.paths.logs_dir,
            self.paths.docs_dir,
            self.outputs.manifest_file,
            self.outputs.train_report_file,
            self.outputs.evaluation_report_file,
        ]


def resolve_inside(base: Path, relative: str, *, label: str) -> Path:
    """解析插件相对路径并拒绝绝对路径、`..` 和符号链接逃逸。"""

    candidate = Path(relative)
    if candidate.is_absolute():
        raise PluginValidationError(
            plugin_name=base.name,
            stage="path_security",
            reason=f"{label} must be relative: {relative}",
            suggested_action="Use a path inside the plugin directory.",
        )
    resolved_base = base.resolve()
    resolved = (resolved_base / candidate).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise PluginValidationError(
            plugin_name=base.name,
            stage="path_security",
            reason=f"{label} escapes plugin directory: {relative}",
            suggested_action="Remove '..' and keep the path inside the plugin directory.",
        )
    return resolved


def load_plugin_manifest(plugin_dir: str | Path) -> ExternalPluginManifest:
    """加载、校验 plugin.yaml 和所有声明路径。"""

    directory = Path(plugin_dir).resolve()
    manifest_path = directory / "plugin.yaml"
    if not manifest_path.is_file():
        raise PluginValidationError(
            plugin_name=directory.name,
            stage="manifest",
            reason=f"plugin.yaml does not exist: {manifest_path}",
            suggested_action="Create plugin.yaml using PLUGIN_SPEC.md.",
        )
    try:
        raw: dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        manifest = ExternalPluginManifest.model_validate(raw)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        error_type = (
            PluginCompatibilityError
            if "api_version" in str(exc) or "schema_version" in str(exc)
            else PluginValidationError
        )
        raise error_type(
            plugin_name=directory.name,
            stage="manifest",
            reason=str(exc),
            suggested_action="Fix the reported plugin.yaml field.",
        ) from exc
    if manifest.plugin.name != directory.name:
        raise PluginValidationError(
            plugin_name=manifest.plugin.name,
            stage="manifest",
            reason=f"manifest name must match directory name {directory.name!r}",
            suggested_action="Rename the directory or plugin.name.",
        )
    for index, relative in enumerate(manifest.relative_paths()):
        resolve_inside(directory, relative, label=f"manifest path #{index + 1}")
    return manifest
