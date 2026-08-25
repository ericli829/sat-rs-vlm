#!/usr/bin/env python3
"""Validate an isolated LAE-DINO/MMDetection sidecar environment."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"unavailable: {exc}"
    return str(getattr(module, "__version__", "ok"))


def _find_configs(source_root: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(source_root.rglob("*.py"))
        if any(token in path.name.lower() for token in ("lae", "dior", "dota"))
        and ("config" in path.parts or "configs" in path.parts)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bert-root", type=Path)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    report: dict[str, Any] = {
        "status": "ok",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "source_root": str(source_root),
        "versions": {
            "torch": _version("torch"),
            "mmcv": _version("mmcv"),
            "mmengine": _version("mmengine"),
            "mmdet": _version("mmdet"),
        },
        "cuda": None,
        "paths": {},
        "discovered_configs": [],
        "errors": [],
    }
    if not source_root.is_dir():
        report["status"] = "failed"
        report["errors"].append(f"source root does not exist: {source_root}")
    else:
        report["discovered_configs"] = _find_configs(source_root)
    if args.discover:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "ok" else 2
    for value, label in (
        (args.config, "config"),
        (args.checkpoint, "checkpoint"),
        (args.bert_root, "bert_root"),
    ):
        if value is None:
            report["errors"].append(f"--{label.replace('_', '-')} is required unless --discover is used")
            continue
        path = value.expanduser().resolve()
        report["paths"][label] = str(path)
        if not path.exists():
            report["errors"].append(f"{label} does not exist: {path}")
    try:
        torch = importlib.import_module("torch")
        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        report["errors"].append(f"torch check failed: {exc}")
    try:
        importlib.import_module("mmdet")
        importlib.import_module("mmengine")
        report["imports"] = {"mmdet": True, "mmengine": True}
    except Exception as exc:
        report["imports"] = {"mmdet": False, "mmengine": False, "error": str(exc)}
        report["errors"].append(f"LAE-DINO import check failed: {exc}")
    if report["errors"]:
        report["status"] = "failed"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

