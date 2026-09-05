"""YAML loading and selected-provider configuration for UHR locators."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.integrations.detectors.config import expand_config_value

from .types import LocatorError


def load_locator_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LocatorError(f"locator config does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LocatorError(f"invalid locator YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise LocatorError("locator config must be a mapping")
    payload["_config_path"] = str(resolved)
    return payload


def selected_provider_config(
    config: Mapping[str, Any],
    *,
    kind: str,
    provider_name: str,
) -> dict[str, Any]:
    profiles = config.get("provider_configs", {})
    if not isinstance(profiles, Mapping):
        raise LocatorError("provider_configs must be a mapping")
    kind_profiles = profiles.get(kind, {})
    if not isinstance(kind_profiles, Mapping):
        raise LocatorError(f"provider_configs.{kind} must be a mapping")
    profile = kind_profiles.get(provider_name, {})
    if not isinstance(profile, Mapping):
        raise LocatorError(f"provider config for {kind}.{provider_name} must be a mapping")
    runtime_section = config.get(kind, {})
    inline = runtime_section.get("config", {}) if isinstance(runtime_section, Mapping) else {}
    if not isinstance(inline, Mapping):
        raise LocatorError(f"{kind}.config must be a mapping")
    merged = {**dict(profile), **dict(inline)}
    try:
        expanded = expand_config_value(merged)
    except Exception as exc:
        raise LocatorError(f"invalid {kind} provider configuration: {exc}") from exc
    return dict(expanded)
