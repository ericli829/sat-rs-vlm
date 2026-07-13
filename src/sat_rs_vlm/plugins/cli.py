"""Public helpers for thin command-line entrypoints shipped by local plugins."""

from __future__ import annotations

from pathlib import Path


def run_local_plugin_command(plugin_dir: Path, command: str) -> int:
    """Delegate a plugin-local command to the main project's explicit runner."""

    from sat_rs_vlm.plugins.runtime import run_external_plugin_from_local_directory

    return run_external_plugin_from_local_directory(plugin_dir, command)
