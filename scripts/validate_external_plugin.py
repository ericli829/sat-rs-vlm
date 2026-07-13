"""验证外部插件结构、入口、依赖、平台和私有导入边界。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sat_rs_vlm.plugins.capability import probe_cuda
from sat_rs_vlm.plugins.dependency import check_requirements
from sat_rs_vlm.plugins.discovery import discover_plugins, resolve_plugin_roots
from sat_rs_vlm.plugins.loader import load_external_plugin
from sat_rs_vlm.plugins.manifest import resolve_inside

PRIVATE_IMPORT = re.compile(r"(?:from|import)\s+sat_rs_vlm\.(?!plugins(?:\.|\s|$))([A-Za-z0-9_.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one external plugin.")
    parser.add_argument("--plugin-root", action="append")
    parser.add_argument("--strategy", required=True)
    return parser.parse_args()


def _forbidden_imports(plugin_dir: Path) -> list[dict[str, Any]]:
    matches = []
    for source in plugin_dir.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if PRIVATE_IMPORT.search(line):
                matches.append({"file": str(source), "line": line_number, "source": line.strip()})
    return matches


def main() -> int:
    args = parse_args()
    roots = resolve_plugin_roots(project_root=PROJECT_ROOT, cli_roots=args.plugin_root)
    plugins = discover_plugins(roots)
    if args.strategy not in plugins:
        print(f"Plugin not found: {args.strategy}")
        return 2
    discovered = plugins[args.strategy]
    manifest = discovered.manifest
    checks: dict[str, Any] = {
        "manifest": True,
        "schema_version": manifest.schema_version,
        "api_version": manifest.plugin.api_version,
        "platform": platform.system().lower(),
    }
    errors = []
    required_paths = {
        "entrypoint": manifest.entrypoint.module,
        "requirements": manifest.dependencies.requirements_file,
        "train_config": manifest.paths.default_train_config,
        "smoke_config": manifest.paths.default_smoke_config,
        "docs": manifest.paths.docs_dir,
        "tests": "tests",
        "checkpoints": manifest.paths.checkpoints_dir,
        "reports": manifest.paths.reports_dir,
        "logs": manifest.paths.logs_dir,
    }
    for label, relative in required_paths.items():
        path = resolve_inside(discovered.directory, relative, label=label)
        checks[label] = path.exists()
        if not path.exists():
            errors.append(f"missing {label}: {path}")
    requirements = resolve_inside(
        discovered.directory,
        manifest.dependencies.requirements_file,
        label="requirements_file",
    )
    statuses = check_requirements(requirements, manifest.plugin.name)
    checks["dependencies"] = [asdict(item) for item in statuses]
    try:
        load_external_plugin(discovered)
        checks["entrypoint_class"] = True
    except Exception as exc:
        checks["entrypoint_class"] = False
        errors.append(str(exc))
    forbidden = _forbidden_imports(discovered.directory)
    checks["forbidden_private_imports"] = forbidden
    if forbidden:
        errors.append("plugin imports private sat_rs_vlm modules")
    allowed_platforms = {item.lower() for item in manifest.compatibility.platforms}
    if platform.system().lower() not in allowed_platforms:
        errors.append(f"platform not supported: {platform.system().lower()}")
    cuda = probe_cuda()
    checks["cuda"] = cuda
    if manifest.compatibility.requires_cuda and not bool(cuda.get("cuda_available")):
        errors.append(f"CUDA required but unavailable: {cuda.get('status')}")
    report = {
        "plugin": manifest.plugin.name,
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }
    output = (
        discovered.directory
        / manifest.paths.reports_dir
        / (f"plugin_validation_{manifest.plugin.name}.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation report: {output}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
