"""Dependency-light locator provider protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sat_rs_vlm.semantics import TaskSpec

from .types import LocatorResult


class LocatorProvider(Protocol):
    provider_name: str

    def locate(self, image_path: Path, query: str | TaskSpec) -> LocatorResult:
        """Locate relevant global-image regions for a question or parsed task."""

    def close(self) -> None:
        """Release detector and retriever resources."""
