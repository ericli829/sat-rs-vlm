"""Small atomic cache for per-query, per-region retrieval scores."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .protocol import RetrievalError

RETRIEVAL_CACHE_SCHEMA_VERSION = "uhr-retrieval-cache-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieval_cache_key(
    *,
    image_path: Path,
    region_xyxy: Sequence[float],
    query: str,
    provider: str,
    model_identity: Any,
    parameters: dict[str, Any],
) -> str:
    resolved = image_path.expanduser().resolve()
    if not resolved.is_file():
        raise RetrievalError(f"retrieval image does not exist: {resolved}")
    stat = resolved.stat()
    payload = {
        "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
        "image_identity": {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _file_sha256(resolved),
        },
        "bbox": [float(value) for value in region_xyxy],
        "query": query.strip(),
        "provider": provider,
        "model_identity": model_identity,
        "parameters": parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RetrievalCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        if not key or any(character not in "0123456789abcdef" for character in key.lower()):
            raise RetrievalError("retrieval cache key must be hexadecimal")
        return self.root / f"{key}.json"

    def get(self, key: str) -> float | None:
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            score = float(payload["score"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"invalid retrieval cache entry: {path}") from exc
        self.hits += 1
        return score

    def put(self, key: str, score: float, metadata: dict[str, Any] | None = None) -> Path:
        path = self._path(key)
        payload = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "score": float(score),
            "metadata": dict(metadata or {}),
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f"{key}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        return path
