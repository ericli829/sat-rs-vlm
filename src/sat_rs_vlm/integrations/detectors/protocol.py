"""Dependency-light proposal provider protocol and coordinate validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

PROPOSAL_SCHEMA_VERSION = "vlm-fo1-proposals-v1"


class ProposalError(RuntimeError):
    """A provider or proposal payload violated the shared protocol."""


class ProposalProvider(Protocol):
    """Provider interface used by precomputation and never by the FO1 model."""

    provider_name: str

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        """Return absolute pixel ``xyxy`` proposals for one image/query."""

    def close(self) -> None:
        """Release a long-lived provider process/model when applicable."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProposalResult:
    """Canonical detector output consumed by the existing precomputed path."""

    boxes_xyxy: list[list[float]]
    scores: list[float]
    latency_ms: float
    provider: str
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.boxes_xyxy) != len(self.scores):
            raise ProposalError("proposal boxes and scores must have equal lengths")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_list": [[float(value) for value in box] for box in self.boxes_xyxy],
            "bbox_scores": [float(value) for value in self.scores],
            "latency_ms": float(self.latency_ms),
            "provider": self.provider,
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
        }


def stable_file_identity(path: str | Path) -> dict[str, Any]:
    """Return a reproducible, cheap identity for a local model/checkpoint."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ProposalError(f"proposal model/checkpoint does not exist: {resolved}")
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _file_sha256(resolved),
        }
    files = []
    aggregate = hashlib.sha256()
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        stat = child.stat()
        relative = str(child.relative_to(resolved))
        digest = _file_sha256(child)
        files.append({"path": relative, "size": stat.st_size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(resolved),
        "file_count": len(files),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "sha256": aggregate.hexdigest(),
    }


def image_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProposalError(f"proposal image does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _file_sha256(resolved),
    }


def proposal_cache_key(
    *,
    provider: str,
    model_identity: Any,
    image: str | Path,
    target_phrase: str,
    parameters: dict[str, Any],
) -> str:
    payload = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "provider": provider,
        "model_identity": model_identity,
        "image_identity": image_identity(image),
        "target_phrase": target_phrase.strip().lower(),
        "parameters": parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonicalize_proposals(
    boxes: Any,
    scores: Any,
    *,
    image_width: int,
    image_height: int,
    coordinate_mode: str = "pixel",
    top_k: int | None = None,
) -> tuple[list[list[float]], list[float], dict[str, int]]:
    """Validate, clamp, deterministically sort, and optionally truncate boxes."""

    if coordinate_mode not in {"pixel", "normalized"}:
        raise ProposalError(f"unsupported proposal coordinate mode: {coordinate_mode}")
    try:
        raw_boxes = [] if boxes is None else list(boxes)
        raw_scores = [] if scores is None else list(scores)
    except TypeError as exc:
        raise ProposalError("proposal boxes and scores must be sequences") from exc
    if len(raw_boxes) != len(raw_scores):
        raise ProposalError(
            f"proposal boxes/scores length mismatch: {len(raw_boxes)} != {len(raw_scores)}"
        )
    if image_width < 1 or image_height < 1:
        raise ProposalError("image dimensions must be positive")

    stats = {"invalid_count": 0, "reordered_count": 0, "clamped_count": 0}
    valid: list[tuple[float, list[float], int]] = []
    # This helper is imported by the Python 3.9 LAE sidecar. Length equality is
    # validated above, so Python 3.10's zip(strict=...) is neither needed nor
    # compatible with that isolated runtime.
    for index, (box, score) in enumerate(
        zip(raw_boxes, raw_scores)  # noqa: B905 - Python 3.9 LAE sidecar
    ):
        try:
            values = [float(value) for value in box]
            score_value = float(score)
        except (TypeError, ValueError):
            stats["invalid_count"] += 1
            continue
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            stats["invalid_count"] += 1
            continue
        if not math.isfinite(score_value):
            stats["invalid_count"] += 1
            continue
        if coordinate_mode == "normalized":
            values = [
                values[0] * image_width,
                values[1] * image_height,
                values[2] * image_width,
                values[3] * image_height,
            ]
        if values[0] > values[2]:
            values[0], values[2] = values[2], values[0]
            stats["reordered_count"] += 1
        if values[1] > values[3]:
            values[1], values[3] = values[3], values[1]
            stats["reordered_count"] += 1
        clamped = [
            min(max(values[0], 0.0), float(image_width)),
            min(max(values[1], 0.0), float(image_height)),
            min(max(values[2], 0.0), float(image_width)),
            min(max(values[3], 0.0), float(image_height)),
        ]
        if clamped != values:
            stats["clamped_count"] += 1
        if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
            stats["invalid_count"] += 1
            continue
        valid.append((score_value, clamped, index))
    valid.sort(key=lambda item: (-item[0], item[2]))
    if top_k is not None:
        if top_k < 0:
            raise ProposalError("top_k must be non-negative")
        valid = valid[:top_k]
    return [item[1] for item in valid], [item[0] for item in valid], stats
