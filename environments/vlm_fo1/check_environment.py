#!/usr/bin/env python3
"""Check VLM-FO1 readiness in either the shared or isolated runtime.

Missing flash-attn, CUDA, the UPN CUDA extension, and an optional UPN
checkpoint are reported as capabilities. They do not fail the shared FO1
path, whose smoke can use precomputed proposal boxes.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.vlm_fo1_loader import (  # noqa: E402
    ensure_official_root,
    resolve_attention_backend,
    validate_model_path,
)


def _value(cli_value: str | None, env_name: str, required: bool) -> Path | None:
    value = (cli_value or os.environ.get(env_name, "")).strip()
    if not value:
        if required:
            raise RuntimeError(f"missing required environment variable: {env_name}")
        return None
    return Path(value).expanduser().resolve()


def _version(module: object) -> str:
    return str(getattr(module, "__version__", "ok"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-mode",
        choices=("official", "shared_rs_vlm"),
        default=os.environ.get("VLM_FO1_RUNTIME_MODE", "shared_rs_vlm"),
    )
    parser.add_argument("--model", default=os.environ.get("VLM_FO1_MODEL", ""))
    parser.add_argument("--root", default=os.environ.get("VLM_FO1_ROOT", ""))
    parser.add_argument(
        "--upn-checkpoint", default=os.environ.get("VLM_FO1_UPN_CHECKPOINT", "")
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default=os.environ.get("VLM_FO1_ATTENTION_BACKEND", "sdpa"),
    )
    parser.add_argument("--allow-missing-upn", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "runtime_mode": args.runtime_mode,
        "environment": "rs-vlm" if args.runtime_mode == "shared_rs_vlm" else "vlm-fo1",
        "attention_backend": args.attention_backend,
        "fo1_model_ready": False,
        "failure_stage": None,
        "imports": {},
        "paths": {},
        "flash_attention_available": False,
        "cuda_available": None,
        "nvcc_available": bool(shutil.which("nvcc")),
        "upn": {
            "python_import": False,
            "python_error": None,
            "cuda_extension": False,
            "cuda_extension_error": None,
            "checkpoint_available": False,
        },
    }
    try:
        if sys.version_info < (3, 10):  # noqa: UP036 - retain runtime guard
            raise RuntimeError("VLM-FO1 requires Python >=3.10")
        model = _value(args.model, "VLM_FO1_MODEL", required=True)
        root = _value(args.root, "VLM_FO1_ROOT", required=True)
        assert model is not None and root is not None
        upn = _value(args.upn_checkpoint, "VLM_FO1_UPN_CHECKPOINT", required=False)
        cache = _value(None, "VLM_FO1_CACHE_DIR", required=False)
        ensure_official_root(root, require_upn=False)
        report["failure_stage"] = "model_path"
        model = validate_model_path(model)
        report["failure_stage"] = None
        report["attention_backend"] = resolve_attention_backend(args.attention_backend)
        if upn is not None:
            report["upn"]["checkpoint_available"] = upn.is_file()  # type: ignore[index]
            if not upn.is_file() and args.runtime_mode == "official" and not args.allow_missing_upn:
                raise RuntimeError(f"VLM_FO1_UPN_CHECKPOINT is missing: {upn}")
        elif args.runtime_mode == "official" and not args.allow_missing_upn:
            raise RuntimeError("VLM_FO1_UPN_CHECKPOINT is required for isolated official mode")
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
            "mmengine",
            "einops",
        ):
            try:
                module = importlib.import_module(module_name)
                if module_name == "torch":
                    torch_module = module
                report["imports"][module_name] = _version(module)  # type: ignore[index]
            except Exception as exc:
                raise RuntimeError(f"failed to import {module_name}: {exc}") from exc

        try:
            flash_module = importlib.import_module("flash_attn")
            report["imports"]["flash_attn"] = _version(flash_module)  # type: ignore[index]
            report["flash_attention_available"] = True
        except Exception as exc:
            report["flash_attention_available"] = False
            report["imports"]["flash_attn"] = f"unavailable: {exc}"  # type: ignore[index]
            if args.runtime_mode == "official":
                raise RuntimeError(
                    "flash_attn is required for legacy official mode; use "
                    "--runtime-mode shared_rs_vlm for the SDPA-compatible path"
                ) from exc

        # Importing the official class first registers the custom config with
        # Transformers; this is required before AutoConfig can read config.json.
        for module_name in ("vlm_fo1", "vlm_fo1.model"):
            try:
                module = importlib.import_module(module_name)
                report["imports"][module_name] = "ok"  # type: ignore[index]
                if module_name == "vlm_fo1.model":
                    model_class = getattr(module, "OmChatQwen25VLForCausalLM", None)
                    report["imports"]["vlm_fo1.model.OmChatQwen25VLForCausalLM"] = (
                        "ok" if model_class is not None else "missing"
                    )  # type: ignore[index]
                    report["vision_towers"] = {
                        "primary_accessor": bool(
                            model_class is not None and hasattr(model_class, "get_vision_tower")
                        ),
                        "auxiliary_accessor": bool(
                            model_class is not None
                            and hasattr(model_class, "get_vision_tower_aux")
                        ),
                    }
            except Exception as exc:
                raise RuntimeError(f"failed to import official module {module_name}: {exc}") from exc
        try:
            from transformers import AutoConfig, AutoTokenizer

            AutoConfig.from_pretrained(str(model), local_files_only=True)
            AutoTokenizer.from_pretrained(str(model), use_fast=False, local_files_only=True)
            report["imports"]["model_config"] = "ok"  # type: ignore[index]
            report["imports"]["tokenizer"] = "ok"  # type: ignore[index]
            report["fo1_model_ready"] = True
        except Exception as exc:
            raise RuntimeError(f"failed to load local FO1 config/tokenizer: {exc}") from exc

        try:
            importlib.import_module("detect_tools.upn")
            report["upn"]["python_import"] = True  # type: ignore[index]
        except Exception as exc:
            report["upn"]["python_error"] = str(exc)  # type: ignore[index]
        if torch_module is not None:
            report["cuda_available"] = bool(torch_module.cuda.is_available())
        extension_errors = []
        try:
            functions = importlib.import_module("detect_tools.upn.ops.functions")
            if getattr(functions, "MSDeformAttnFunction", None) is None:
                raise RuntimeError("MSDeformAttnFunction is unavailable")
            if not report["cuda_available"]:
                raise RuntimeError("CUDA is unavailable, so the compiled extension is unusable")
            report["upn"]["cuda_extension"] = True  # type: ignore[index]
        except Exception as exc:
            extension_errors.append(f"detect_tools.upn.ops.functions: {exc}")
            report["upn"]["cuda_extension_error"] = "; ".join(extension_errors)  # type: ignore[index]
        report["upn"]["ready_for_inference"] = bool(
            report["upn"]["python_import"]
            and report["upn"]["cuda_extension"]
            and report["upn"]["checkpoint_available"]
        )  # type: ignore[index]
        print(json.dumps({"status": "ok", **report}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    **report,
                    "status": "failed",
                    "failure_stage": report.get("failure_stage") or "environment",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
