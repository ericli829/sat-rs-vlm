#!/usr/bin/env python3
"""Isolated JSONL worker for official VisRAG-Ret region scoring."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.retrievers.visrag import (  # noqa: E402
    VisRAGDirectRetrieverProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


def score_request(
    provider: VisRAGDirectRetrieverProvider,
    request: dict[str, Any],
) -> dict[str, Any]:
    for field in ("id", "image", "query", "regions_xyxy"):
        if field not in request:
            raise ValueError(f"missing request field: {field}")
    # Reserve stdout for one-response-per-line JSON. Third-party remote code
    # may print model-loading diagnostics; the sidecar stderr log remains
    # available for intentional diagnostics.
    with contextlib.redirect_stdout(io.StringIO()):
        result = provider.score_regions(
            Path(str(request["image"])),
            str(request["query"]),
            request["regions_xyxy"],
        )
    return {"id": request["id"], "status": "ok", "result": result.to_dict()}


def main() -> int:
    args = parse_args()
    try:
        config = json.loads(args.config_json)
        if not isinstance(config, dict):
            raise ValueError("worker config must be an object")
        config["runtime"] = "direct"
        provider = VisRAGDirectRetrieverProvider(config)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "id": None,
                    "status": "failed",
                    "failure_stage": "model_config",
                    "error": str(exc),
                }
            ),
            flush=True,
        )
        return 2
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request: Any = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                response = score_request(provider, request)
            except Exception as exc:
                response = {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "status": "failed",
                    "failure_stage": "retrieval_scoring",
                    "error": str(exc),
                }
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
