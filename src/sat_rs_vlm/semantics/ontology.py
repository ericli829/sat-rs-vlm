"""Remote-sensing ontology loading shared by runtime and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ontology(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"semantic ontology does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid semantic ontology JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("semantic ontology must be a JSON object")
    for field in ("ontology_version", "objects", "relations", "changes"):
        if field not in payload:
            raise ValueError(f"semantic ontology is missing {field}")
    if not all(
        isinstance(payload[field], dict) for field in ("objects", "relations", "changes")
    ):
        raise ValueError("objects, relations and changes must be JSON objects")
    return payload
