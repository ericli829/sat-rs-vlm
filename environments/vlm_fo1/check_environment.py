#!/usr/bin/env python3
"""Validate the isolated VLM-FO1 environment without importing rs-vlm."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _env_path(name: str, required: bool) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        if required:
            raise RuntimeError(f"missing required environment variable: {name}")
        return None
    return Path(value).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-upn", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "environment": "vlm-fo1",
        "imports": {},
        "paths": {},
        "flash_attention_available": False,
        "cuda_available": None,
        "nvcc_available": bool(shutil.which("nvcc")),
    }
    try:
        if sys.version_info < (3, 10):  # noqa: UP036 - retain runtime guard
            raise RuntimeError("VLM-FO1 requires Python >=3.10")
        model = _env_path("VLM_FO1_MODEL", required=True)
        root = _env_path("VLM_FO1_ROOT", required=True)
        upn = _env_path("VLM_FO1_UPN_CHECKPOINT", required=not args.allow_missing_upn)
        cache = _env_path("VLM_FO1_CACHE_DIR", required=False)
        assert model is not None and root is not None
        root = root.resolve()
        if not root.is_dir():
            raise RuntimeError(f"VLM_FO1_ROOT is not a directory: {root}")
        for relative in ("vlm_fo1", "detect_tools", "detect_tools/upn"):
            if not (root / relative).is_dir():
                raise RuntimeError(
                    f"VLM_FO1_ROOT is missing required directory: {root / relative}"
                )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if not (model / "config.json").is_file():
            raise RuntimeError(f"VLM_FO1_MODEL is missing config.json: {model}")
        if upn is not None and not upn.is_file():
            raise RuntimeError(f"VLM_FO1_UPN_CHECKPOINT is missing: {upn}")
        report["paths"] = {
            "VLM_FO1_ROOT": str(root),
            "VLM_FO1_MODEL": str(model),
            "VLM_FO1_UPN_CHECKPOINT": str(upn) if upn else None,
            "VLM_FO1_CACHE_DIR": str(cache) if cache else None,
        }
        torch_module = None
        for module_name in (
            "torch",
            "torchvision",
            "transformers",
            "timm",
            "accelerate",
            "mmengine",
        ):
            try:
                module = importlib.import_module(module_name)
                if module_name == "torch":
                    torch_module = module
                report["imports"][module_name] = getattr(module, "__version__", "ok")  # type: ignore[index]
            except Exception as exc:
                raise RuntimeError(f"failed to import {module_name}: {exc}") from exc
        try:
            flash_module = importlib.import_module("flash_attn")
            report["imports"]["flash_attn"] = getattr(flash_module, "__version__", "ok")  # type: ignore[index]
            report["flash_attention_available"] = True
        except Exception as exc:
            report["flash_attention_available"] = False
            raise RuntimeError(
                "flash_attn is required because the official builder uses "
                f"flash_attention_2: {exc}"
            ) from exc
        for module_name in ("detect_tools.upn", "vlm_fo1"):
            try:
                importlib.import_module(module_name)
                report["imports"][module_name] = "ok"  # type: ignore[index]
            except Exception as exc:
                raise RuntimeError(
                    f"failed to import official module {module_name}: {exc}"
                ) from exc
        if torch_module is not None:
            report["cuda_available"] = bool(torch_module.cuda.is_available())
        print(json.dumps({"status": "ok", **report}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error": str(exc), **report}, ensure_ascii=False, indent=2
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
