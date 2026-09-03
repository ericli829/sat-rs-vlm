"""列出显式本地根目录中的外部插件。"""

# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.plugins.capability import validate_platform_capability
from sat_rs_vlm.plugins.dependency import check_requirements
from sat_rs_vlm.plugins.discovery import discover_plugins, resolve_plugin_roots
from sat_rs_vlm.plugins.loader import load_external_plugin
from sat_rs_vlm.plugins.manifest import resolve_inside

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List external fine-tuning plugins.")
    parser.add_argument("--plugin-root", action="append")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--show-incompatible", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = resolve_plugin_roots(project_root=PROJECT_ROOT, cli_roots=args.plugin_root)
    plugins = discover_plugins(roots)
    rows: list[dict[str, Any]] = []
    for name, discovered in sorted(plugins.items()):
        manifest = discovered.manifest
        requirements = resolve_inside(
            discovered.directory,
            manifest.dependencies.requirements_file,
            label="requirements_file",
        )
        statuses = check_requirements(requirements, name)
        row: dict[str, Any] = {
            "name": name,
            "version": manifest.plugin.version,
            "status": manifest.plugin.status,
            "description": manifest.plugin.description,
            "api_version": manifest.plugin.api_version,
            "requires_cuda": manifest.compatibility.requires_cuda,
            "dependencies_status": [asdict(item) for item in statuses],
            "default_config": str(discovered.directory / manifest.paths.default_train_config),
            "plugin_path": str(discovered.directory),
        }
        if args.validate:
            try:
                load_external_plugin(discovered)
                row["entrypoint_valid"] = True
            except Exception as exc:
                row["entrypoint_valid"] = False
                row["validation_error"] = str(exc)
        if args.validate or args.show_incompatible:
            try:
                row["compatibility"] = validate_platform_capability(manifest)
                row["compatible"] = True
            except Exception as exc:
                row["compatible"] = False
                row["compatibility_error"] = str(exc)
        rows.append(row)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            dependencies = (
                ",".join(
                    f"{item['package']}:{item['status']}" for item in row["dependencies_status"]
                )
                or "none"
            )
            compatibility = "unknown" if "compatible" not in row else str(row["compatible"]).lower()
            print(
                f"{row['name']} {row['version']} [{row['status']}] "
                f"api={row['api_version']} cuda={row['requires_cuda']} "
                f"dependencies={dependencies} compatible={compatibility} "
                f"config={row['default_config']} path={row['plugin_path']}"
            )
            if row.get("compatibility_error"):
                print(f"  compatibility_error={row['compatibility_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
