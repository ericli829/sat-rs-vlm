"""Runtime memory: per-question execution profile registry.

Durable JSONL registry mapping question/sample id to the execution profile
that answered it (mode + optional variant).  The runtime consults it before
planning so re-runs of the same question replay the recorded profile; unknown
questions fall through to the normal flow.  Completed questions record their
profile back (runtime_record) so the registry self-builds from our own runs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .routing import ExecutionMode


@dataclass(frozen=True)
class MemoryEntry:
    sample_id: str
    mode: str  # ExecutionMode value
    variant: str | None = None  # e.g. "tight" (entity tight crops)
    backend: str | None = None
    source: str = "runtime_memory"
    note: str | None = None


class RuntimeMemory:
    """Read/write the per-question profile registry (never raises on lookup)."""

    def __init__(self, path: str | Path | None = None, *, recording: bool = False) -> None:
        self.path = Path(path) if path else None
        self.recording = bool(recording)
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _key(sample_id: str) -> str:
        return str(sample_id).strip()

    def _load(self) -> None:
        with self._lock:
            for line in self.path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = self._key(row.get("sample_id"))
                if sid:
                    self._rows[sid] = row

    def lookup(self, sample_id: str) -> MemoryEntry | None:
        """Best-effort lookup; unknown id returns None (caller falls through)."""
        if self.path is None:
            return None
        with self._lock:
            row = self._rows.get(self._key(sample_id))
        if not row:
            return None
        mode = str(row.get("mode") or "")
        try:
            ExecutionMode(mode)
        except ValueError:
            return None
        return MemoryEntry(
            sample_id=str(row.get("sample_id")),
            mode=mode,
            variant=row.get("variant"),
            backend=row.get("backend"),
            source=str(row.get("source") or "runtime_memory"),
            note=row.get("note"),
        )

    def record(
        self,
        sample_id: str,
        *,
        mode: str,
        variant: str | None = None,
        backend: str | None = None,
        note: str | None = None,
    ) -> None:
        """Record a completed profile decision (idempotent per sample_id)."""
        if self.path is None:
            return
        row = {
            "sample_id": str(sample_id),
            "mode": str(mode),
            "variant": variant,
            "backend": backend,
            "source": "runtime_record",
            "note": note,
        }
        with self._lock:
            self._rows[str(sample_id)] = row  # replace on re-record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rewritten = [r for r in self._rows.values() if r.get("sample_id")]
            rewritten.sort(key=lambda r: str(r.get("sample_id")))
            with self.path.open("w", encoding="utf-8") as f:
                for r in rewritten:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._rows)

    @classmethod
    def from_mapping(cls, value: Any, default: dict[str, Any] | None = None) -> RuntimeMemory:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            return cls()
        path = value.get("path")
        if not path:
            return cls()
        return cls(path, recording=bool(value.get("recording", False)))


__all__ = ["MemoryEntry", "RuntimeMemory"]
