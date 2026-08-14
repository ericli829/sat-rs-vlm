"""Stable extension points for deployment detection and recovery chains.

Consensus is intentionally an interface only. The project does not claim that an
untested generalist/specialist ensemble improves reliability.
"""

from __future__ import annotations

from typing import Any, Protocol


class OutputDetector(Protocol):
    def validate(self, output: Any, *, context: dict[str, Any]) -> dict[str, Any]: ...


class ConsensusDetector(Protocol):
    def compare(
        self,
        outputs: list[Any],
        *,
        context: dict[str, Any],
    ) -> dict[str, Any]: ...


class RecoveryAction(Protocol):
    def recover(self, report: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]: ...
