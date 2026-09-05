"""Shared configuration/path normalization for detector integrations."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import ProposalError

_UNRESOLVED_ENV_RE = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def expand_config_value(value: Any) -> Any:
    """Recursively expand environment variables and reject unresolved ones."""

    if isinstance(value, Mapping):
        return {str(key): expand_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config_value(item) for item in value)
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        unresolved = _UNRESOLVED_ENV_RE.search(expanded)
        if unresolved:
            raise ProposalError(
                f"unresolved detector configuration environment variable: {unresolved.group(0)}"
            )
        return expanded
    return value


def resolve_config_path(value: Any, *, label: str) -> Path:
    """Expand one configured path before constructing a ``Path`` object."""

    expanded = expand_config_value(value)
    if expanded is None or str(expanded).strip() == "":
        raise ProposalError(f"{label} must be configured")
    path = Path(str(expanded)).expanduser().resolve()
    return path
