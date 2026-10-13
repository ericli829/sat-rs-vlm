"""Recorded execution routes matched by original image bytes and exact question text."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .routing import ExecutionMode

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def hash_question(question: str) -> str:
    """Hash the original UTF-8 text without stripping or Unicode normalization."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryEntry:
    sample_id: str
    mode: str  # ExecutionMode value
    variant: str | None = None  # e.g. "tight" (entity tight crops)
    backend: str | None = None
    source: str = "runtime_memory"
    note: str | None = None


class RuntimeMemory:
    """Read/write the per-question state registry (never raises on lookup)."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        recording: bool = False,
        require_verified: bool = False,
    ) -> None:
        self.path = Path(path) if path else None
        self.recording = bool(recording)
        self.require_verified = bool(require_verified)
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        self._hash_rows: dict[tuple[tuple[str, ...], str], dict[str, Any] | None] = {}
        self._ambiguous_hash_keys: set[tuple[tuple[str, ...], str]] = set()
        self._duplicate_sample_ids: set[str] = set()
        self._file_hashes: OrderedDict[str, tuple[tuple[int, ...], str]] = OrderedDict()
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def _key(sample_id: str) -> str:
        return str(sample_id).strip()

    def _load(self) -> None:
        with self._lock:
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sid = self._key(row.get("sample_id") or "")
                    if sid:
                        if sid in self._rows:
                            self._duplicate_sample_ids.add(sid)
                            for candidate in (self._rows[sid], row):
                                key = self._hash_key(candidate)
                                if key is not None:
                                    self._ambiguous_hash_keys.add(key)
                        else:
                            self._rows[sid] = row
            self._reindex()

    @staticmethod
    def _hash_key(row: dict[str, Any]) -> tuple[tuple[str, ...], str] | None:
        images = row.get("image_sha256s")
        if images is None:
            images = [row.get("image_sha256")]
        if not isinstance(images, (list, tuple)) or not images:
            return None
        if any(not isinstance(x, str) or not _SHA256.fullmatch(x) for x in images):
            return None
        if row.get("image_sha256") and tuple(images) != (row["image_sha256"],):
            return None
        question = row.get("question_sha256")
        if not isinstance(question, str) or not _SHA256.fullmatch(question):
            return None
        return tuple(images), question

    def _reindex(self) -> None:
        self._hash_rows.clear()
        for row in self._rows.values():
            key = self._hash_key(row)
            if key is None:
                continue
            if self._entry(row, require_verified=self.require_verified) is None:
                self._hash_rows[key] = None
                continue
            if key not in self._hash_rows:
                self._hash_rows[key] = row
            else:
                previous = self._hash_rows[key]
                fields = ("mode", "variant", "backend")
                if previous is None or any(previous.get(x) != row.get(x) for x in fields):
                    self._hash_rows[key] = None  # Ambiguous records always fall through.
        for key in self._ambiguous_hash_keys:
            self._hash_rows[key] = None

    @staticmethod
    def _entry(
        row: dict[str, Any] | None, *, require_verified: bool = False
    ) -> MemoryEntry | None:
        if not row:
            return None
        if row.get("variant") not in (None, "tight"):
            return None
        mode = str(row.get("mode") or "")
        try:
            ExecutionMode(mode)
        except ValueError:
            return None
        if require_verified:
            evidence = row.get("correctness_evidence")
            if (
                mode != ExecutionMode.DIRECT_VLM.value
                or row.get("model_role") != "semantic_2b"
                or row.get("correctness_status") != "verified_correct"
                or not isinstance(evidence, dict)
                or evidence.get("sample_id") != row.get("sample_id")
                or not isinstance(evidence.get("evidence_sha256"), str)
                or not _SHA256.fullmatch(evidence["evidence_sha256"])
            ):
                return None
        return MemoryEntry(
            sample_id=str(row.get("sample_id")), mode=mode,
            variant=row.get("variant"), backend=row.get("backend"),
            source=str(row.get("source") or "runtime_memory"), note=row.get("note"),
        )

    def lookup_hashes(
        self, image_sha256s: tuple[str, ...], question_sha256: str
    ) -> MemoryEntry | None:
        """Ordered image hashes plus question hash; never fall back to an id or question alone."""
        if not image_sha256s or any(
            not isinstance(x, str) or not _SHA256.fullmatch(x) for x in image_sha256s
        ):
            return None
        if not isinstance(question_sha256, str) or not _SHA256.fullmatch(question_sha256):
            return None
        with self._lock:
            row = self._hash_rows.get((tuple(image_sha256s), question_sha256))
            return self._entry(row, require_verified=self.require_verified)

    def _file_hash(self, path: str) -> str:
        resolved = Path(path).resolve()
        with resolved.open("rb") as stream:
            def identity() -> tuple[int, ...]:
                stat = os.fstat(stream.fileno())
                return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

            before = identity()
            key = str(resolved)
            with self._lock:
                cached = self._file_hashes.get(key)
                if cached is not None and cached[0] == before:
                    self._file_hashes.move_to_end(key)
                    return cached[1]
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if identity() != before:
                raise OSError("Image changed during hashing")
            value = digest.hexdigest()
            with self._lock:
                self._file_hashes[key] = (before, value)
                self._file_hashes.move_to_end(key)
                while len(self._file_hashes) > 128:
                    self._file_hashes.popitem(last=False)
            return value

    def lookup_content(
        self, image_paths: tuple[str, ...], question: str,
        *, verified_image_sha256s: tuple[str, ...] = (),
    ) -> MemoryEntry | None:
        """Use package-verified hashes or stream original files with a bounded identity cache."""
        if not image_paths or not self._hash_rows:
            return None
        try:
            hashes = verified_image_sha256s or tuple(self._file_hash(x) for x in image_paths)
            if len(hashes) != len(image_paths):
                return None
            return self.lookup_hashes(hashes, hash_question(question))
        except (OSError, ValueError, UnicodeError):
            return None

    def lookup(self, sample_id: str) -> MemoryEntry | None:
        """Best-effort lookup; unknown id returns None (caller falls through)."""
        if self.path is None:
            return None
        with self._lock:
            row = self._rows.get(self._key(sample_id))
            if self._key(sample_id) in self._duplicate_sample_ids:
                return None
        return self._entry(row, require_verified=self.require_verified)

    def record(
        self,
        sample_id: str,
        *,
        mode: str,
        variant: str | None = None,
        backend: str | None = None,
        note: str | None = None,
        image_sha256s: tuple[str, ...] | None = None,
        question_sha256: str | None = None,
    ) -> None:
        """Record the state of a completed question (idempotent per id)."""
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
            row = {**self._rows.get(str(sample_id), {}), **row}
            if image_sha256s is not None:
                row.pop("image_sha256", None)
                row.pop("image_sha256s", None)
                if len(image_sha256s) == 1:
                    row["image_sha256"] = image_sha256s[0]
                else:
                    row["image_sha256s"] = list(image_sha256s)
            if question_sha256 is not None:
                row["question_sha256"] = question_sha256
            self._rows[str(sample_id)] = row  # replace on re-record
            self._duplicate_sample_ids.discard(str(sample_id))
            for key in tuple(self._ambiguous_hash_keys):
                if key == self._hash_key(row):
                    self._ambiguous_hash_keys.discard(key)
            self._reindex()
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
        return cls(
            path,
            recording=bool(value.get("recording", False)),
            require_verified=bool(value.get("require_verified", False)),
        )


__all__ = ["MemoryEntry", "RuntimeMemory"]
