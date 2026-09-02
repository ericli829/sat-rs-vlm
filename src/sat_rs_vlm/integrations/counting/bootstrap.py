"""Ensure the workspace counting_system package is importable from TaskGraph."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def counting_system_src() -> Path:
    env = os.environ.get("COUNTING_SYSTEM_SRC")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    worktree = here.parents[4]
    nested = worktree / "04_counting_system_plan" / "src"
    if nested.exists():
        return nested
    sibling = worktree.parent / "04_counting_system_plan" / "src"
    return sibling


def ensure_counting_system_importable() -> Path:
    src = counting_system_src()
    src_str = str(src)
    if src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src
