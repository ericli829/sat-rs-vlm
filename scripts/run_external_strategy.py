"""显式运行普通本地目录中的外部微调插件。"""

# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.plugins.capability import validate_platform_capability
from sat_rs_vlm.plugins.dependency import (
    check_requirements,
    install_missing_requirements,
    parse_requirements,
    write_dependency_report,
)
from sat_rs_vlm.plugins.discovery import discover_plugins, resolve_plugin_roots
from sat_rs_vlm.plugins.errors import ExternalPluginError, PluginDependencyError
from sat_rs_vlm.plugins.loader import load_external_plugin
from sat_rs_vlm.plugins.manifest import resolve_inside
from sat_rs_vlm.plugins.runtime import (
    apply_external_overrides,
    execute_external_plugin,
    load_external_config,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an external fine-tuning plugin.")
    parser.add_argument("--plugin-root", action="append")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config")
    parser.add_argument("--model-dir")
    parser.add_argument("--processor-dir")
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--install-missing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--wheel-dir")
    parser.add_argument("--allow-non-venv-install", action="store_true")
    parser.add_argument("--python-executable")
    return parser.parse_args()


def _delegate_python(args: argparse.Namespace) -> int | None:
    if not args.python_executable:
        return None
    executable = Path(args.python_executable).resolve()
    if executable == Path(sys.executable).resolve():
        return None
    forwarded = []
    skip_next = False
    for item in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if item == "--python-executable":
            skip_next = True
            continue
        forwarded.append(item)
    command = [str(executable), str(Path(__file__).resolve()), *forwarded]
    print("Delegating external plugin command:", " ".join(command))
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: getattr(args, name)
        for name in (
            "model_dir",
            "processor_dir",
            "train_file",
            "val_file",
            "image_root",
            "max_train_samples",
            "max_eval_samples",
            "max_steps",
            "resume_from_checkpoint",
            "skip_eval",
        )
    }


def main() -> int:
    args = parse_args()
    delegated = _delegate_python(args)
    if delegated is not None:
        return delegated
    if args.strategy.lower() == "lora":
        print("LoRA is the stable built-in baseline. Use scripts/train_qwen3vl_lora.py.")
        return 2
    try:
        roots = resolve_plugin_roots(
            project_root=PROJECT_ROOT,
            cli_roots=args.plugin_root,
        )
        plugins = discover_plugins(roots)
        if args.strategy not in plugins:
            available = ", ".join(sorted(plugins)) or "<none>"
            raise KeyError(f"External strategy {args.strategy!r} not found. Available: {available}")
        discovered = plugins[args.strategy]
        manifest = discovered.manifest
        validate_platform_capability(manifest)
        requirements_file = resolve_inside(
            discovered.directory,
            manifest.dependencies.requirements_file,
            label="requirements_file",
        )
        statuses = check_requirements(requirements_file, manifest.plugin.name)
        install_command = None
        unsatisfied = [item for item in statuses if item.status != "satisfied"]
        install_allowed = bool(args.install_missing) or os.getenv(
            "SAT_RS_VLM_PLUGIN_ALLOW_INSTALL", ""
        ).lower() in {"1", "true", "yes"}
        offline = bool(args.offline) or os.getenv("SAT_RS_VLM_PLUGIN_OFFLINE", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        if unsatisfied and install_allowed:
            install_command = install_missing_requirements(
                plugin_name=manifest.plugin.name,
                requirements_file=requirements_file,
                statuses=statuses,
                offline=offline,
                wheel_dir=Path(args.wheel_dir) if args.wheel_dir else None,
                allow_non_venv_install=bool(args.allow_non_venv_install),
            )
            statuses = check_requirements(requirements_file, manifest.plugin.name)
            unsatisfied = [item for item in statuses if item.status != "satisfied"]
        dependency_report = (
            discovered.directory / manifest.paths.reports_dir / "dependency_report.json"
        )
        write_dependency_report(dependency_report, statuses, install_command)
        if unsatisfied:
            details = ", ".join(
                f"{item.package}: required {item.required_version or '*'}, "
                f"current {item.current_version}, status {item.status}"
                for item in unsatisfied
            )
            suggestion = (
                f"Run {sys.executable} -m pip install -r {requirements_file} in a compatible venv, "
                "or explicitly use --install-missing."
            )
            raise PluginDependencyError(
                plugin_name=manifest.plugin.name,
                stage="dependency_check",
                reason=details,
                suggested_action=suggestion,
            )
        if args.check_only:
            print(f"Dependency check passed: {dependency_report}")
            return 0
        plugin = load_external_plugin(discovered)
        config_path = (
            Path(args.config).resolve()
            if args.config
            else resolve_inside(
                discovered.directory,
                manifest.paths.default_train_config,
                label="default_train_config",
            )
        )
        config = apply_external_overrides(load_external_config(config_path), _overrides(args))
        requirement_names = {
            requirement.name.lower()
            for requirement in parse_requirements(requirements_file, manifest.plugin.name)
        }
        report = execute_external_plugin(
            discovered=discovered,
            plugin=plugin,
            config=config,
            project_root=PROJECT_ROOT,
            output_dir=args.output_dir,
            dry_run=bool(args.dry_run),
            forward_only=bool(args.forward_only),
            skip_eval=bool(args.skip_eval),
            install_missing=install_allowed,
            require_bitsandbytes="bitsandbytes" in requirement_names,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ExternalPluginError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
