"""外部插件依赖版本检查与显式安装。"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

from sat_rs_vlm.plugins.errors import PluginDependencyError


@dataclass(frozen=True)
class DependencyStatus:
    requirement: str
    package: str
    required_version: str
    current_version: str | None
    status: str


def parse_requirements(path: Path, plugin_name: str) -> list[Requirement]:
    """解析常见 PEP 508 requirement；拒绝 pip option、URL 和 editable 行。"""

    requirements = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("-") or " @ " in line:
            raise PluginDependencyError(
                plugin_name=plugin_name,
                stage="dependency_check",
                reason=f"unsupported requirements line {line_number}: {raw}",
                suggested_action="Use package names with normal version specifiers.",
            )
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise PluginDependencyError(
                plugin_name=plugin_name,
                stage="dependency_check",
                reason=f"invalid requirement on line {line_number}: {raw}",
                suggested_action="Fix requirements.txt syntax.",
            ) from exc
        if requirement.marker is None or requirement.marker.evaluate():
            requirements.append(requirement)
    return requirements


def check_requirements(path: Path, plugin_name: str) -> list[DependencyStatus]:
    """使用 distribution metadata 检查缺失与版本冲突。"""

    if not path.is_file():
        raise PluginDependencyError(
            plugin_name=plugin_name,
            stage="dependency_check",
            reason=f"requirements file does not exist: {path}",
            suggested_action="Create the requirements file declared in plugin.yaml.",
        )
    statuses = []
    for requirement in parse_requirements(path, plugin_name):
        try:
            current = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            current = None
        if current is None:
            status = "missing"
        elif current not in requirement.specifier:
            status = "version_conflict"
        else:
            status = "satisfied"
        statuses.append(
            DependencyStatus(
                requirement=str(requirement),
                package=requirement.name,
                required_version=str(requirement.specifier),
                current_version=current,
                status=status,
            )
        )
    return statuses


def in_virtual_environment() -> bool:
    """判断当前解释器是否位于 venv/virtualenv。"""

    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def install_missing_requirements(
    *,
    plugin_name: str,
    requirements_file: Path,
    statuses: list[DependencyStatus],
    offline: bool,
    wheel_dir: Path | None,
    allow_non_venv_install: bool,
) -> list[str]:
    """仅安装缺失包；现有版本冲突默认拒绝，避免隐式降级。"""

    conflicts = [item for item in statuses if item.status == "version_conflict"]
    if conflicts:
        details = ", ".join(
            f"{item.package} required {item.required_version}, current {item.current_version}"
            for item in conflicts
        )
        raise PluginDependencyError(
            plugin_name=plugin_name,
            stage="dependency_install",
            reason=f"version conflict requires environment changes: {details}",
            suggested_action="Create a plugin-specific virtual environment; no downgrade was run.",
        )
    missing = [item for item in statuses if item.status == "missing"]
    if not missing:
        return []
    if not in_virtual_environment() and not allow_non_venv_install:
        raise PluginDependencyError(
            plugin_name=plugin_name,
            stage="dependency_install",
            reason="current Python is not a virtual environment",
            suggested_action="Activate .venv or pass --allow-non-venv-install explicitly.",
        )
    command = [sys.executable, "-m", "pip", "install"]
    if offline:
        command.append("--no-index")
        if wheel_dir is not None:
            command.extend(["--find-links", str(wheel_dir.resolve())])
    command.extend(["-r", str(requirements_file.resolve())])
    print("Dependency install command:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise PluginDependencyError(
            plugin_name=plugin_name,
            stage="dependency_install",
            reason=f"pip exited with code {completed.returncode}",
            suggested_action="Inspect pip output; no automatic rollback or downgrade is performed.",
        )
    return command


def write_dependency_report(
    path: Path,
    statuses: list[DependencyStatus],
    install_command: list[str] | None = None,
) -> None:
    """保存依赖检查和可选安装命令。"""

    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "in_virtual_environment": in_virtual_environment(),
        "dependencies": [asdict(item) for item in statuses],
        "install_command": install_command,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
