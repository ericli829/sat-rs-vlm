from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime import CountResult, to_jsonable


class TraceWriter:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "trace.jsonl"

    def record(self, result: CountResult, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "count": result.count,
            "num_detections": len(result.detections),
            "detections": [asdict(d) for d in result.detections],
            "provenance": _strip_raw(result.provenance),
        }
        if extra:
            payload["extra"] = extra
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def _strip_raw(provenance: dict[str, Any]) -> dict[str, Any]:
    out = dict(provenance)
    raw = out.get("raw_proposals")
    if isinstance(raw, list):
        out["raw_proposals"] = len(raw)
        out["raw_proposals_preview"] = raw[:20]
    return out
