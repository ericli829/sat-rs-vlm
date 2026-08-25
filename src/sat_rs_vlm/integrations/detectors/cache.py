"""Small JSON proposal cache with deterministic keys and atomic writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .protocol import ProposalError, ProposalResult


class ProposalCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        if not key or any(character not in "0123456789abcdef" for character in key.lower()):
            raise ProposalError("proposal cache key must be hexadecimal")
        return self.root / f"{key}.json"

    def get(self, key: str) -> ProposalResult | None:
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = ProposalResult(
                boxes_xyxy=payload["bbox_list"],
                scores=payload["bbox_scores"],
                latency_ms=payload["latency_ms"],
                provider=payload["provider"],
                model_id=payload["model_id"],
                metadata=payload.get("metadata", {}),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.misses += 1
            raise ProposalError(f"invalid proposal cache entry: {path}") from exc
        self.hits += 1
        return result

    def put(self, key: str, result: ProposalResult) -> Path:
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

