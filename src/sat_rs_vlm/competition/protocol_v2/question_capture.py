"""Explicit private question capture, separate from ordinary service logs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .models import PredictRequest


class QuestionCapture:
    def __init__(self, path: str | Path):
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._stream = target.open("a", encoding="utf-8", newline="\n")

    def record(self, request: PredictRequest) -> None:
        # Only validated wire fields. Headers, credentials and image bytes are absent.
        entry = {
            "schema_version": 1,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "question_sha256": hashlib.sha256(request.question.encode("utf-8")).hexdigest(),
            "request": request.model_dump(mode="json", exclude_none=True),
        }
        self._stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()
