"""解析 configs/paths.yaml 与环境变量。业务代码不写死 AutoDL 绝对路径。

AutoDL 上逻辑根是 /root/autodl-fs，大文件可落到 /root/autodl-tmp。
本机若没有这两个目录，则映射到仓库上一级的 autodl-fs / autodl-tmp，
这样 Windows 开发机和 AutoDL 读同一份 configs/paths.yaml。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PACKAGE_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
PATHS_CONFIG = CONFIG_DIR / "paths.yaml"
PROMPT_CONFIG = CONFIG_DIR / "prompt_profiles.yaml"

LINUX_FS = "/root/autodl-fs"
LINUX_TMP = "/root/autodl-tmp"

ENV_MAP = {
    "COUNTING_DATASET_ROOT": ("datasets", "xlrsbench_lite"),
    "COUNTING_LAE_ROOT": ("models", "lae_dino_root"),
    "COUNTING_LAE_WEIGHTS": ("models", "lae_dino_weights"),
    "COUNTING_LAE_REPO": ("models", "lae_dino_repo"),
    "COUNTING_GEORS_CKPT": ("models", "georsclip_ckpt"),
    "COUNTING_GEORS_DIR": ("models", "georsclip_dir"),
    "HF_ENDPOINT": ("huggingface", "endpoint"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _set_nested(tree: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cur: dict[str, Any] = tree
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _has_xlrs(root: Path) -> bool:
    return (root / "datasets" / "xlrsbench-lite" / "counting.jsonl").exists()


def host_roots(workspace: Path | None = None) -> tuple[Path, Path]:
    """返回 (autodl_fs, autodl_tmp)。优先环境变量，其次已落盘的 XLRS 数据，否则 Linux 实盘 / 仓库旁目录。"""

    root = workspace or PACKAGE_ROOT.parent
    fs_env = os.environ.get("COUNTING_AUTODL_FS") or os.environ.get("AUTODL_FS")
    tmp_env = os.environ.get("COUNTING_AUTODL_TMP") or os.environ.get("AUTODL_TMP")
    linux_fs = Path(LINUX_FS)
    linux_tmp = Path(LINUX_TMP)
    workspace_fs = root / "autodl-fs"
    workspace_tmp = root / "autodl-tmp"
    if fs_env:
        fs = Path(fs_env).expanduser()
    elif linux_fs.exists() and _has_xlrs(linux_fs):
        fs = linux_fs
    elif _has_xlrs(workspace_fs):
        fs = workspace_fs
    elif linux_fs.exists():
        fs = linux_fs
    else:
        fs = workspace_fs
    if tmp_env:
        tmp = Path(tmp_env).expanduser()
    elif linux_tmp.exists():
        tmp = linux_tmp
    else:
        tmp = workspace_tmp
    return fs, tmp


def _replace_prefix(value: str, mapping: list[tuple[str, Path]]) -> str:
    ordered = sorted(mapping, key=lambda item: len(item[0]), reverse=True)
    for old, new in ordered:
        old = old.rstrip("/\\")
        if value == old or value.startswith(old + "/") or value.startswith(old + "\\"):
            rest = value[len(old) :].lstrip("/\\")
            return str((new / rest).resolve()) if rest else str(new.resolve())
    return value


def relocate_autodl_paths(tree: dict[str, Any], *, workspace: Path | None = None) -> dict[str, Any]:
    fs, tmp = host_roots(workspace)
    mapping = [(LINUX_FS, fs), (LINUX_TMP, tmp)]

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: walk(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        if isinstance(obj, str):
            return _replace_prefix(obj, mapping)
        return obj

    relocated = walk(tree)
    relocated["autodl_fs"] = str(fs.resolve())
    relocated["autodl_tmp"] = str(tmp.resolve())
    return relocated


def apply_huggingface_env(paths: dict[str, Any] | None = None) -> None:
    hf = (paths or {}).get("huggingface") or {}
    # AutoDL / Cursor 沙箱常注入 OFFLINE=1，真实检测必须允许从镜像拉权重
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("HF_ENDPOINT", str(hf.get("endpoint") or "https://hf-mirror.com"))
    home = hf.get("home")
    if home:
        os.environ.setdefault("HF_HOME", str(home))
        os.environ.setdefault("HF_HUB_CACHE", str(Path(home) / "hub"))
    if hf.get("disable_xet", True):
        os.environ["HF_HUB_DISABLE_XET"] = "1"


@lru_cache(maxsize=4)
def load_paths() -> dict[str, Any]:
    data = relocate_autodl_paths(_read_yaml(PATHS_CONFIG))
    for env_name, keys in ENV_MAP.items():
        value = os.environ.get(env_name)
        if value:
            _set_nested(data, keys, value)
    apply_huggingface_env(data)
    return data


@lru_cache(maxsize=4)
def load_default_config() -> dict[str, Any]:
    return _read_yaml(DEFAULT_CONFIG)


@lru_cache(maxsize=1)
def load_prompt_profiles() -> dict[str, Any]:
    return _read_yaml(PROMPT_CONFIG)


def load_config(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deep_merge({"paths": load_paths()}, load_default_config())
    if extra:
        cfg = deep_merge(cfg, extra)
    return cfg


def resolve_existing(*candidates: str | Path | None) -> Path | None:
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser()
        if path.exists():
            return path.resolve()
    return None


def dataset_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["datasets"]["xlrsbench_lite"])


def lae_weights(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["models"]["lae_dino_weights"])


def lae_repo(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["models"]["lae_dino_repo"])


def georsclip_ckpt(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["models"]["georsclip_ckpt"])


def bert_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["models"]["bert_dir"])
