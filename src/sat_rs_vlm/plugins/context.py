"""外部插件可见的只读运行上下文。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PublicService = Callable[..., Any]


@dataclass(frozen=True)
class PluginContext:
    """只暴露文档承诺的路径、模式与公共服务。"""

    project_root: Path
    plugin_root: Path
    plugin_dir: Path
    output_dir: Path
    model_dir: Path
    processor_dir: Path
    train_file: Path
    val_file: Path | None
    image_root: Path
    device: str
    dry_run: bool
    forward_only: bool
    install_missing: bool
    common_services: Mapping[str, PublicService]

    def service(self, name: str) -> PublicService:
        """读取已公开服务；未知名称立即失败。"""

        try:
            return self.common_services[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.common_services))
            raise KeyError(
                f"Unknown public plugin service {name!r}. Available: {available}"
            ) from exc
