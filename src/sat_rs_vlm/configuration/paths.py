"""集中式路径配置与用途感知的目录校验。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from sat_rs_vlm.configuration.environment import ENV_PATTERN, expand_environment

WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def resolve_path_value(value: str | Path, *, base_dir: Path) -> Path:
    """展开用户目录并把相对路径解析到指定基准目录。

    Windows 盘符路径在 Windows 上由 `Path` 原生处理；在 POSIX 上保留为
    Windows 风格路径对象的文本，避免错误拼接到项目根目录。
    """

    text = str(value)
    if ENV_PATTERN.search(text):
        raise ValueError(f"Unresolved environment variable in path: {text}")
    expanded = Path(text).expanduser()
    if expanded.is_absolute():
        return expanded
    if WINDOWS_ABSOLUTE.match(text):
        return Path(str(PureWindowsPath(text)))
    return (base_dir / expanded).resolve()


class PathConfig(BaseModel):
    """项目所有数据、模型、输出和缓存路径的统一配置。"""

    model_config = ConfigDict(protected_namespaces=())

    project_root: Path
    dataset_root: Path | None = None
    model_root: Path | None = None
    output_root: Path
    cache_root: Path
    temp_root: Path
    tensorboard_root: Path
    backup_root: Path
    hf_home: Path
    hf_hub_cache: Path
    torch_home: Path
    pip_cache_dir: Path

    ENV_TO_FIELD: ClassVar[dict[str, str]] = {
        "PROJECT_ROOT": "project_root",
        "DATA_ROOT": "dataset_root",
        "MODEL_ROOT": "model_root",
        "OUTPUT_ROOT": "output_root",
        "CACHE_ROOT": "cache_root",
        "TMPDIR": "temp_root",
        "TENSORBOARD_ROOT": "tensorboard_root",
        "BACKUP_ROOT": "backup_root",
        "HF_HOME": "hf_home",
        "HF_HUB_CACHE": "hf_hub_cache",
        "TORCH_HOME": "torch_home",
        "PIP_CACHE_DIR": "pip_cache_dir",
    }

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        project_root: Path,
        environ: Mapping[str, str] | None = None,
        apply_environment_overrides: bool = True,
    ) -> PathConfig:
        """从配置映射构建路径，并按需应用路径环境变量覆盖。"""

        env = dict(os.environ if environ is None else environ)
        raw = dict(values)
        raw.setdefault("project_root", str(project_root))
        if apply_environment_overrides:
            for env_name, field in cls.ENV_TO_FIELD.items():
                if env_name in env:
                    raw[field] = env[env_name]
        raw = expand_environment(raw, environ=env, allow_unresolved=False)
        resolved_project = resolve_path_value(raw["project_root"], base_dir=project_root)

        defaults: dict[str, str] = {
            "output_root": "outputs",
            "cache_root": ".cache",
            "temp_root": ".tmp",
            "tensorboard_root": "outputs/tensorboard",
            "backup_root": "outputs/backups",
            "hf_home": ".cache/huggingface",
            "hf_hub_cache": ".cache/huggingface/hub",
            "torch_home": ".cache/torch",
            "pip_cache_dir": ".cache/pip",
        }
        normalized: dict[str, Path | None] = {"project_root": resolved_project}
        for field in cls.model_fields:
            if field == "project_root":
                continue
            item = raw.get(field, defaults.get(field))
            normalized[field] = (
                resolve_path_value(item, base_dir=resolved_project) if item is not None else None
            )
        return cls.model_validate(normalized)

    def validate_inputs(
        self,
        *,
        require_dataset: bool = True,
        require_model: bool = True,
    ) -> None:
        """验证输入路径；绝不创建不存在的数据或模型目录。"""

        checks: list[tuple[str, Path | None, bool]] = [
            ("dataset_root", self.dataset_root, require_dataset),
            ("model_root", self.model_root, require_model),
        ]
        for name, path, required in checks:
            if not required:
                continue
            if path is None:
                raise FileNotFoundError(f"Required input path is not configured: {name}")
            if not path.is_dir():
                raise FileNotFoundError(f"Required input directory does not exist: {name}={path}")

    def create_output_directories(self) -> None:
        """创建允许由程序管理的输出和缓存目录。"""

        for path in (
            self.output_root,
            self.cache_root,
            self.temp_root,
            self.tensorboard_root,
            self.backup_root,
            self.hf_home,
            self.hf_hub_cache,
            self.torch_home,
            self.pip_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def as_environment(self) -> dict[str, str]:
        """返回可传递给子进程的标准路径环境变量。"""

        result: dict[str, str] = {}
        for env_name, field in self.ENV_TO_FIELD.items():
            value = getattr(self, field)
            if value is not None:
                result[env_name] = str(value)
        return result
