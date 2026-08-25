#!/usr/bin/env python3
"""Precompute detector proposals without sending reference answers to providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.detectors.cache import ProposalCache  # noqa: E402
from sat_rs_vlm.integrations.detectors.config import (  # noqa: E402
    expand_config_value,
    resolve_config_path,
)
from sat_rs_vlm.integrations.detectors.lae_dino_sidecar import (  # noqa: E402
    PINNED_LAE_DINO_SOURCE_REVISION,
)
from sat_rs_vlm.integrations.detectors.protocol import (  # noqa: E402
    ProposalError,
    proposal_cache_key,
    stable_file_identity,
)
from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider  # noqa: E402
from sat_rs_vlm.integrations.vlm_fo1 import extract_count_target_phrase  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(payload)
    return rows


def _user_question_image(row: Mapping[str, Any]) -> tuple[str, str]:
    """Read only user content; assistant/reference content is intentionally ignored."""

    question = str(row.get("question", "")).strip()
    image = str(row.get("image", "")).strip()
    messages = row.get("messages")
    if not isinstance(messages, list):
        return question, image
    for message in messages:
        if not isinstance(message, Mapping) or str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and not question:
            question = content.strip()
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("type", "")).lower()
            if kind == "text" and not question:
                question = str(item.get("text", "")).strip()
            if kind not in {"image", "image_url"} or image:
                continue
            value = item.get("image")
            if isinstance(value, Mapping):
                value = value.get("url")
            if value is None:
                value = item.get("image_url")
                if isinstance(value, Mapping):
                    value = value.get("url")
            image = str(value or "").strip()
    if not question or not image:
        raise ValueError(f"row {row.get('id')}: user question/image is missing")
    return question, image


def _resolve_image(image: str, image_root: Path | None) -> Path:
    path = Path(image).expanduser()
    if path.is_absolute() or image_root is None:
        return path.resolve()
    candidate = (image_root / path).resolve()
    if candidate.is_file():
        return candidate
    if image_root.name.lower() == "vrsbench" and path.parts and path.parts[0].lower() == "vrsbench":
        return (image_root.parent / path).resolve()
    return candidate


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"proposal config must be a mapping: {path}")
    return payload


def _value(args: argparse.Namespace, section: Mapping[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return section.get(name, default)


def _build_provider_config(args: argparse.Namespace, proposal: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(proposal)
    for name in (
        "model_path",
        "source_root",
        "config_path",
        "checkpoint",
        "bert_root",
        "worker_python",
        "worker_script",
        "device",
        "dtype",
        "box_threshold",
        "text_threshold",
        "score_threshold",
        "top_k",
        "nms_threshold",
    ):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    return dict(expand_config_value(config))


def _manifest_model_identity(provider_name: str, provider_config: Mapping[str, Any]) -> Any:
    """Capture only assets that affect detector semantics, not source trees."""

    if str(provider_name).startswith("lae_dino"):
        checkpoint = resolve_config_path(provider_config.get("checkpoint"), label="checkpoint")
        config_path = resolve_config_path(
            provider_config.get("config_path") or provider_config.get("config"),
            label="config_path",
        )
        bert_root = resolve_config_path(provider_config.get("bert_root"), label="bert_root")
        return {
            "provider": provider_name,
            "checkpoint": stable_file_identity(checkpoint),
            "config": stable_file_identity(config_path),
            "bert": stable_file_identity(bert_root),
            "source_revision": provider_config.get(
                "source_revision", PINNED_LAE_DINO_SOURCE_REVISION
            ),
            "checkpoint_training_regime": provider_config.get(
                "checkpoint_training_regime", "unspecified"
            ),
            "inference_query_mode": provider_config.get(
                "inference_query_mode", "target_conditioned_text_prompt"
            ),
        }
    model_path = provider_config.get("model_path") or provider_config.get("checkpoint")
    return (
        stable_file_identity(resolve_config_path(model_path, label="model_path"))
        if model_path
        else None
    )


def precompute_rows(
    rows: list[dict[str, Any]],
    *,
    provider_name: str,
    provider_config: Mapping[str, Any],
    image_root: Path | None = None,
    cache: ProposalCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = create_proposal_provider(provider_name, provider_config)
    output_rows: list[dict[str, Any]] = []
    counts = {"ok": 0, "unsupported": 0, "failed": 0}
    try:
        model_identity = getattr(provider, "model_identity", dict(provider_config))
        cache_parameters = {
            key: provider_config.get(key)
            for key in (
                "box_threshold",
                "text_threshold",
                "score_threshold",
                "top_k",
                "nms_threshold",
                "dtype",
            )
        }
        for row in rows:
            output = dict(row)
            try:
                question, image_text = _user_question_image(row)
                target = extract_count_target_phrase(question)
                output["target_phrase"] = target.phrase
                output["target_status"] = target.status
                if not target.supported:
                    counts["unsupported"] += 1
                    output.update(
                        {
                            "bbox_list": [],
                            "bbox_scores": [],
                            "proposal_latency_ms": 0.0,
                            "proposal_provider": provider_name,
                            "proposal_metadata": {
                                "schema_version": "vlm-fo1-proposal-row-v1",
                                "status": "unsupported",
                                "target_reason": target.reason,
                            },
                        }
                    )
                    output_rows.append(output)
                    continue
                image_path = _resolve_image(image_text, image_root)
                key = proposal_cache_key(
                    provider=provider_name,
                    model_identity=model_identity,
                    image=image_path,
                    target_phrase=target.phrase or "",
                    parameters=cache_parameters,
                )
                result = cache.get(key) if cache is not None else None
                cache_hit = result is not None
                if result is None:
                    result = provider.predict(image_path, target.phrase or "")
                    if cache is not None:
                        cache.put(key, result)
                counts["ok"] += 1
                metadata = dict(result.metadata)
                metadata.update(
                    {
                        "schema_version": "vlm-fo1-proposal-row-v1",
                        "status": "ok",
                        "provider": result.provider,
                        "model_id": result.model_id,
                        "cache_key": key,
                        "cache_hit": cache_hit,
                    }
                )
                output.update(
                    {
                        "bbox_list": result.boxes_xyxy,
                        "bbox_scores": result.scores,
                        "proposal_latency_ms": result.latency_ms,
                        "proposal_provider": result.provider,
                        "proposal_model": result.model_id,
                        "proposal_metadata": metadata,
                    }
                )
            except Exception as exc:
                counts["failed"] += 1
                failure_stage = str(getattr(exc, "failure_stage", "proposal_generation"))
                output.update(
                    {
                        "bbox_list": [],
                        "bbox_scores": [],
                        "proposal_latency_ms": 0.0,
                        "proposal_provider": provider_name,
                        "proposal_metadata": {
                            "schema_version": "vlm-fo1-proposal-row-v1",
                            "status": "failed",
                            "failure_stage": failure_stage,
                            "error": str(exc),
                        },
                    }
                )
            output_rows.append(output)
    finally:
        provider.close()
    return output_rows, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--provider",
        choices=("mock", "grounding_dino", "lae_dino_lae1m", "lae_dino_dior", "lae_dino_dota"),
    )
    parser.add_argument("--model-path")
    parser.add_argument("--source-root")
    parser.add_argument("--config-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--bert-root")
    parser.add_argument("--worker-python")
    parser.add_argument("--worker-script")
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--box-threshold", type=float)
    parser.add_argument("--text-threshold", type=float)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = _load_config(args.config)
        proposal = payload.get("proposal", {})
        if not isinstance(proposal, Mapping):
            raise ValueError("proposal config must be a mapping")
        provider_name = _value(args, proposal, "provider", None)
        evaluation = payload.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            evaluation = {}
        input_path = args.input or payload.get("input") or evaluation.get("input")
        output_path = args.output or payload.get("output") or proposal.get("precomputed_file")
        if not provider_name or not input_path or not output_path:
            raise ValueError("--provider, --input, and --output are required")
        input_path = resolve_config_path(input_path, label="proposal input")
        output_path = resolve_config_path(output_path, label="proposal output")
        if output_path.exists() and not args.force:
            raise FileExistsError(f"output exists; pass --force to overwrite: {output_path}")
        data_section = payload.get("data", {})
        if not isinstance(data_section, Mapping):
            data_section = {}
        image_root_value = (
            args.image_root or payload.get("image_root") or data_section.get("image_root")
        )
        image_root = (
            resolve_config_path(image_root_value, label="proposal image_root")
            if image_root_value
            else None
        )
        provider_config = _build_provider_config(args, proposal)
        cache_dir = _value(args, proposal, "cache_dir", None)
        cache = (
            ProposalCache(resolve_config_path(cache_dir, label="proposal cache_dir"))
            if cache_dir
            else None
        )
        all_rows = _read_jsonl(input_path)
        if args.max_samples is not None and args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        rows = all_rows[: args.max_samples] if args.max_samples is not None else all_rows
        output_rows, counts = precompute_rows(
            rows,
            provider_name=str(provider_name),
            provider_config=provider_config,
            image_root=image_root,
            cache=cache,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in output_rows
            ),
            encoding="utf-8",
        )
        manifest_path = args.manifest or output_path.with_suffix(
            output_path.suffix + ".manifest.json"
        )
        manifest = {
            "schema_version": "vlm-fo1-proposal-manifest-v1",
            "provider": provider_name,
            "provider_config": provider_config,
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "input_rows": len(rows),
            "input_rows_total": len(all_rows),
            "counts": counts,
            "cache": {
                "root": str(cache.root) if cache else None,
                "hits": cache.hits if cache else 0,
                "misses": cache.misses if cache else 0,
            },
            "model_identity": _manifest_model_identity(provider_name, provider_config),
            "reference_not_sent_to_provider": True,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(output_path),
                    "manifest": str(manifest_path),
                    "counts": counts,
                }
            )
        )
        return 0
    except (OSError, ValueError, ProposalError, RuntimeError) as exc:
        print(f"proposal precomputation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
