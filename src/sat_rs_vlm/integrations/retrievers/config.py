"""Environment-aware configuration helpers for retriever integrations."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import RetrievalError

_UNRESOLVED_ENV_RE = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def expand_config_value(value: Any) -> Any:
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
            raise RetrievalError(
                "unresolved retriever configuration environment variable: "
                f"{unresolved.group(0)}"
            )
        return expanded
    return value


def resolve_config_path(value: Any, *, label: str) -> Path:
    expanded = expand_config_value(value)
    if expanded is None or str(expanded).strip() == "":
        raise RetrievalError(f"{label} must be configured")
    return Path(str(expanded)).expanduser().resolve()
