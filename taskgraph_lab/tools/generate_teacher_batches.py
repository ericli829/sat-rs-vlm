from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from taskgraph_lab import PROMPT_VERSION, SCHEMA_VERSION
from taskgraph_lab.generation.batch_generation import generate_teacher_batch
from taskgraph_lab.generation.generate import (
    RateLimiter,
    RuntimeSettings,
    append_jsonl,
    iter_samples,
    load_completed_ids,
)
from taskgraph_lab.generation.provider import provider_from_config
from taskgraph_lab.tools.summarize import read_jsonl, summarize

LAB_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("batch generation config must be a YAML mapping")
    return payload


def _manifest(
    *,
    input_path: Path,
    config_path: Path,
    system_prompt_path: Path,
    batch_contract_path: Path,
    batch_size: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    provider_config = dict(config.get("provider") or {})
    return {
        "version": "taskgraph-stage1-batch-generation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "system_prompt_sha256": _sha256(system_prompt_path),
        "batch_contract_path": str(batch_contract_path.resolve()),
        "batch_contract_sha256": _sha256(batch_contract_path),
        "batch_size": batch_size,
        "provider": provider_config.get("type"),
        "model": provider_config.get("model"),
        "thinking": provider_config.get("thinking"),
        "reasoning_effort": provider_config.get("reasoning_effort"),
    }


def _validate_or_write_manifest(path: Path, current: dict[str, Any]) -> None:
    identity_fields = {
        key: current[key]
        for key in (
            "version",
            "schema_version",
            "prompt_version",
            "input_sha256",
            "config_sha256",
            "system_prompt_sha256",
            "batch_contract_sha256",
            "batch_size",
            "provider",
            "model",
            "thinking",
            "reasoning_effort",
        )
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_identity = {key: previous.get(key) for key in identity_fields}
        if previous_identity != identity_fields:
            raise ValueError(
                "existing stage1 output provenance does not match current generation settings"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_call_records(path: Path, calls: list[Any]) -> None:
    for call in calls:
        append_jsonl(
            path,
            {
                "call_id": call.call_id,
                "kind": call.kind,
                "batch_id": call.batch_id,
                "sample_ids": call.sample_ids,
                "trace": call.trace,
                "transport": call.transport,
                "error": call.error,
            },
        )


def run_stage1(
    *,
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    system_prompt_path: Path,
    batch_contract_path: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    settings = RuntimeSettings.from_mapping(dict(config.get("runtime") or {}))
    batch_config = dict(config.get("batch") or {})
    batch_size = int(batch_config.get("teacher_batch_size", 4))
    if batch_size < 1:
        raise ValueError("batch.teacher_batch_size must be >= 1")
    manifest = _manifest(
        input_path=input_path,
        config_path=config_path,
        system_prompt_path=system_prompt_path,
        batch_contract_path=batch_contract_path,
        batch_size=batch_size,
        config=config,
    )
    _validate_or_write_manifest(output_dir / "run_manifest.json", manifest)

    raw_path = output_dir / "raw.jsonl"
    paths = {kind: output_dir / f"{kind}.jsonl" for kind in ("valid", "repaired", "rejected")}
    calls_path = output_dir / "calls.jsonl"
    completed = load_completed_ids(raw_path)
    samples = [sample for sample in iter_samples(input_path) if sample.sample_id not in completed]
    provider = provider_from_config(dict(config.get("provider") or {}))
    limiter = RateLimiter(settings.requests_per_minute)
    total_input_samples = sum(1 for _ in iter_samples(input_path))
    submitted = 0

    def progress(event: dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "event": "stage1_started",
                "total_samples": total_input_samples,
                "already_completed": len(completed),
                "remaining": len(samples),
                "batch_size": batch_size,
                "thinking": manifest["thinking"],
                "reasoning_effort": manifest["reasoning_effort"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for offset in range(0, len(samples), batch_size):
        subset = samples[offset : offset + batch_size]
        result = generate_teacher_batch(
            subset,
            provider=provider,
            limiter=limiter,
            settings=settings,
            system_prompt=system_prompt_path.read_text(encoding="utf-8"),
            batch_transport_contract=batch_contract_path.read_text(encoding="utf-8"),
            batch_size=batch_size,
            teacher_batch_max_input_tokens=int(
                batch_config.get("teacher_batch_max_input_tokens", 24000)
            ),
            teacher_batch_max_samples=int(
                batch_config.get("teacher_batch_max_samples", batch_size)
            ),
            max_transport_retries=int(batch_config.get("max_transport_retries", 1)),
            progress=progress,
        )
        for outcome in result.outcomes:
            append_jsonl(raw_path, outcome.raw)
            if outcome.destination and outcome.record is not None:
                append_jsonl(paths[outcome.destination], outcome.record)
            submitted += 1
        _append_call_records(calls_path, result.calls)
        print(
            json.dumps(
                {
                    "event": "stage1_batch_persisted",
                    "submitted_this_run": submitted,
                    "remaining_this_run": len(samples) - submitted,
                    "total_samples": total_input_samples,
                }
            ),
            flush=True,
        )

    report = summarize(
        read_jsonl(raw_path),
        read_jsonl(paths["valid"]),
        read_jsonl(paths["repaired"]),
        read_jsonl(paths["rejected"]),
        [],
    )
    final_report = {
        "output_dir": str(output_dir.resolve()),
        "batch_size": batch_size,
        "thinking": manifest["thinking"],
        "reasoning_effort": manifest["reasoning_effort"],
        **report,
    }
    (output_dir / "generation_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "stage1_completed", **final_report}, ensure_ascii=False), flush=True)
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable TaskGraph batch generation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--system-prompt", type=Path, default=LAB_ROOT / "prompts/system_prompt.txt"
    )
    parser.add_argument(
        "--batch-contract",
        type=Path,
        default=LAB_ROOT / "prompts/batch_transport_contract.txt",
    )
    args = parser.parse_args()
    run_stage1(
        input_path=args.input,
        config_path=args.config,
        output_dir=args.output_dir,
        system_prompt_path=args.system_prompt,
        batch_contract_path=args.batch_contract,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
