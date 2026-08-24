#!/usr/bin/env python3
"""Evaluate VRSBench counting through an isolated VLM-FO1 JSONL worker.

The rs-vlm interpreter only orchestrates JSONL.  The worker process is run by
``VLM_FO1_PYTHON`` and owns every official FO1/UPN import.  ``--backend mock``
is intentionally available for deterministic unit and protocol smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.counting_protocol import (  # noqa: E402
    summarize_exact_cardinality_counting,
)
from sat_rs_vlm.evaluation.parsers import parse_count  # noqa: E402
from sat_rs_vlm.integrations.vlm_fo1 import (  # noqa: E402
    FO1_PROMPT_PROFILES,
    extract_count_target_phrase,
    prediction_count_text,
)

DEFAULT_ECOUNT = PROJECT_ROOT / "data/evaluation/tiers_v2/e_count_v2.jsonl"
DEFAULT_FULL = PROJECT_ROOT / "data/processed/multisource/vrsbench_levircc_eval_full.jsonl"
DEFAULT_WORKER = PROJECT_ROOT / "scripts/integrations/vlm_fo1_worker.py"
DEFAULT_AUDIT = PROJECT_ROOT / "reports/integrations/vlm_fo1_source_audit.json"


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
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(payload)
    return rows


def _message_parts(row: Mapping[str, Any]) -> tuple[str, str, str]:
    question = str(row.get("question", "")).strip()
    image = str(row.get("image", "")).strip()
    reference = str(row.get("reference", "")).strip()
    for message in row.get("messages", []) if isinstance(row.get("messages"), list) else []:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if isinstance(content, str):
            if role == "user" and not question:
                question = content.strip()
            elif role == "assistant" and not reference:
                reference = content.strip()
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("type", "")).lower()
            if kind in {"image", "image_url"} and not image:
                image_value = item.get("image")
                if isinstance(image_value, Mapping):
                    image_value = image_value.get("url")
                if image_value is None:
                    image_value = item.get("image_url")
                    if isinstance(image_value, Mapping):
                        image_value = image_value.get("url")
                image = str(image_value or "").strip()
            if kind == "text" and role == "user" and not question:
                question = str(item.get("text", "")).strip()
        if role == "assistant" and not reference:
            text_values = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and str(item.get("type", "")) == "text"
            ]
            reference = " ".join(value for value in text_values if value).strip()
    if not question:
        raise ValueError(f"row {row.get('id')}: counting question is missing")
    if not reference:
        raise ValueError(f"row {row.get('id')}: counting reference is missing")
    if not image:
        raise ValueError(f"row {row.get('id')}: image path is missing")
    return image, question, reference


def _resolve_image(image: str, image_root: Path | None) -> str:
    path = Path(image).expanduser()
    if path.is_absolute() or image_root is None:
        return str(path)
    candidate = image_root / path
    if candidate.is_file():
        return str(candidate)
    if image_root.name.lower() == "vrsbench" and path.parts and path.parts[0].lower() == "vrsbench":
        return str(image_root.parent / path)
    return str(candidate)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
    }


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"FO1 config must be a mapping: {path}")
    return payload


def _config_value(config: Mapping[str, Any], section: str, name: str, fallback: Any) -> Any:
    section_value = config.get(section, {})
    if isinstance(section_value, Mapping) and name in section_value:
        value = section_value[name]
    else:
        value = config.get(name, fallback)
    return os.path.expandvars(value) if isinstance(value, str) else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scope", choices=("e_count_v2", "full_vrsbench_quantity"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--backend", choices=("official", "mock"))
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--worker-script", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--upn-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--proposal-score-threshold", type=float)
    parser.add_argument("--proposal-top-k", type=int)
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--prompt-profile", choices=FO1_PROMPT_PROFILES)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    scope = args.scope or _config_value(config, "evaluation", "scope", "e_count_v2")
    input_default = DEFAULT_ECOUNT if scope == "e_count_v2" else DEFAULT_FULL
    input_path = args.input or _config_value(config, "evaluation", "input", str(input_default))
    output_default = PROJECT_ROOT / "reports/evaluation/vlm_fo1" / str(scope)
    settings: dict[str, Any] = {
        "scope": scope,
        "input": Path(input_path),
        "expected_population": _config_value(config, "evaluation", "expected_population", None),
        "output_dir": Path(
            args.output_dir or _config_value(config, "evaluation", "output_dir", output_default)
        ),
        "image_root": args.image_root
        or Path(
            _config_value(config, "data", "image_root", os.environ.get("VLM_FO1_IMAGE_ROOT", ""))
        )
        if (
            args.image_root
            or _config_value(config, "data", "image_root", os.environ.get("VLM_FO1_IMAGE_ROOT", ""))
        )
        else None,
        "max_samples": args.max_samples
        if args.max_samples is not None
        else _config_value(config, "evaluation", "max_samples", None),
        "backend": args.backend or _config_value(config, "worker", "backend", "official"),
        "worker_python": args.worker_python
        or Path(
            _config_value(
                config, "worker", "python", os.environ.get("VLM_FO1_PYTHON", sys.executable)
            )
        ),
        "worker_script": args.worker_script
        or Path(_config_value(config, "worker", "script", DEFAULT_WORKER)),
        "model": args.model
        or Path(_config_value(config, "model", "path", os.environ.get("VLM_FO1_MODEL", ""))),
        "upn_checkpoint": args.upn_checkpoint
        or Path(
            _config_value(
                config, "proposal", "checkpoint", os.environ.get("VLM_FO1_UPN_CHECKPOINT", "")
            )
        ),
        "device": args.device or _config_value(config, "model", "device", "cuda"),
        "proposal_score_threshold": args.proposal_score_threshold
        if args.proposal_score_threshold is not None
        else _config_value(config, "proposal", "score_threshold", 0.3),
        "proposal_top_k": args.proposal_top_k
        if args.proposal_top_k is not None
        else _config_value(config, "proposal", "top_k", 100),
        "nms_threshold": args.nms_threshold
        if args.nms_threshold is not None
        else _config_value(config, "proposal", "nms_threshold", 0.8),
        "max_new_tokens": args.max_new_tokens
        if args.max_new_tokens is not None
        else _config_value(config, "generation", "max_new_tokens", 4096),
        "temperature": args.temperature
        if args.temperature is not None
        else _config_value(config, "generation", "temperature", 0.0),
        "top_p": args.top_p
        if args.top_p is not None
        else _config_value(config, "generation", "top_p", 0.05),
        "prompt_profile": args.prompt_profile
        or _config_value(config, "generation", "prompt_profile", "official_fo1"),
        "audit": args.audit or Path(_config_value(config, "provenance", "audit", DEFAULT_AUDIT)),
    }
    for key in (
        "input",
        "output_dir",
        "image_root",
        "worker_python",
        "worker_script",
        "model",
        "upn_checkpoint",
        "audit",
    ):
        value = settings[key]
        if isinstance(value, Path) and not value.is_absolute():
            settings[key] = PROJECT_ROOT / value
    if settings["max_samples"] is not None and int(settings["max_samples"]) < 1:
        raise ValueError("--max-samples must be positive")
    if settings["scope"] not in {"e_count_v2", "full_vrsbench_quantity"}:
        raise ValueError(f"unsupported scope: {settings['scope']}")
    return settings


def _select_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "e_count_v2":
        selected = [row for row in rows if str(row.get("task_type", "")).lower() == "counting"]
    else:
        candidates = [
            row
            for row in rows
            if str(row.get("task_type", "")).lower() == "counting"
            and isinstance(row.get("metadata"), Mapping)
            and str(row["metadata"].get("dataset", "")).lower() == "vrsbench"
            and str(row["metadata"].get("qa_type", "")).lower() == "object quantity"
        ]
        # The public VRSBench Quantity replay population is 6,131 rows: it
        # excludes non-numeric answer strings while formal metrics still
        # apply exact-cardinality eligibility downstream.
        selected = []
        for row in candidates:
            _, _, reference = _message_parts(row)
            if parse_count(reference).value is not None:
                selected.append(row)
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError("selected counting row has empty id")
        if sample_id in seen:
            raise ValueError(f"duplicate selected sample id: {sample_id}")
        if str(row.get("task_type", "")).lower() != "counting":
            raise ValueError(f"scope row is not counting: {sample_id}")
        seen.add(sample_id)
        output.append(row)
    return output


def _worker_command(settings: Mapping[str, Any]) -> list[str]:
    command = [
        str(settings["worker_python"]),
        str(settings["worker_script"]),
        "--backend",
        str(settings["backend"]),
        "--model",
        str(settings["model"]),
        "--upn-checkpoint",
        str(settings["upn_checkpoint"]),
        "--device",
        str(settings["device"]),
        "--proposal-score-threshold",
        str(settings["proposal_score_threshold"]),
        "--proposal-top-k",
        str(settings["proposal_top_k"]),
        "--nms-threshold",
        str(settings["nms_threshold"]),
        "--max-new-tokens",
        str(settings["max_new_tokens"]),
        "--temperature",
        str(settings["temperature"]),
        "--top-p",
        str(settings["top_p"]),
        "--prompt-profile",
        str(settings["prompt_profile"]),
    ]
    return command


def _run_worker(
    requests: list[dict[str, Any]], settings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    process = subprocess.Popen(
        _worker_command(settings),
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    responses: list[dict[str, Any]] = []
    for request in requests:
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            error = process.stderr.read() if process.stderr is not None else ""
            responses.append(
                {
                    "id": request["id"],
                    "status": "failed",
                    "failure_stage": "worker_process",
                    "error": error.strip() or "worker exited before returning a response",
                }
            )
            break
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "id": request["id"],
                "status": "failed",
                "failure_stage": "worker_protocol",
                "error": f"worker emitted invalid JSON: {exc}",
            }
        if not isinstance(response, dict):
            response = {
                "id": request["id"],
                "status": "failed",
                "failure_stage": "worker_protocol",
                "error": "worker response must be an object",
            }
        response.setdefault("id", request["id"])
        responses.append(response)
    process.stdin.close()
    process.wait(timeout=120)
    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.returncode not in {0, None} and len(responses) < len(requests):
        for request in requests[len(responses) :]:
            responses.append(
                {
                    "id": request["id"],
                    "status": "failed",
                    "failure_stage": "worker_process",
                    "error": stderr.strip() or f"worker exit code {process.returncode}",
                }
            )
    return responses


def _diagnostics(responses: list[dict[str, Any]]) -> dict[str, Any]:
    supported = sum(response.get("target_status") == "supported" for response in responses)
    proposal_counts = [
        float(response["proposal_count_raw"])
        for response in responses
        if isinstance(response.get("proposal_count_raw"), (int, float))
    ]
    selected_counts = [
        float(len(response.get("selected_region_indexes", [])))
        for response in responses
        if isinstance(response.get("selected_region_indexes", []), list)
    ]
    latencies = [
        float(response.get("upn_latency_ms", 0.0)) + float(response.get("fo1_latency_ms", 0.0))
        for response in responses
        if response.get("upn_latency_ms") is not None and response.get("fo1_latency_ms") is not None
    ]
    failures = [response for response in responses if response.get("status") == "failed"]
    return {
        "target_phrase_supported_rate": supported / len(responses) if responses else None,
        "target_phrase_supported_count": supported,
        "target_phrase_total_count": len(responses),
        "proposal_count": _numeric_summary(proposal_counts),
        "selected_region_count": _numeric_summary(selected_counts),
        "proposal_failure_count": sum(
            str(response.get("failure_stage", "")).startswith("proposal") for response in failures
        ),
        "fo1_parse_failure_count": sum(
            response.get("failure_stage") == "fo1_parse" for response in failures
        ),
        "worker_failure_count": sum(
            response.get("failure_stage") in {"worker_process", "worker_protocol", "backend_init"}
            for response in failures
        ),
        "latency_ms": _numeric_summary(latencies),
        "status_counts": {
            status: sum(response.get("status") == status for response in responses)
            for status in sorted({str(response.get("status")) for response in responses})
        },
        "region_selection_text_count_disagreements": sum(
            response.get("status") == "ok" and response.get("fo1_count_agrees_with_text") is False
            for response in responses
        ),
    }


def _summary_markdown(
    metrics: Mapping[str, Any], diagnostics: Mapping[str, Any], provenance: Mapping[str, Any]
) -> str:
    overall = dict(metrics.get("overall", {}))
    proposal_counts = diagnostics.get("proposal_count", {})
    selected_counts = diagnostics.get("selected_region_count", {})
    lines = [
        "# VLM-FO1 VRSBench counting evaluation",
        "",
        f"- scope: `{provenance.get('scope')}`",
        f"- prompt profile: `{provenance.get('prompt_profile')}`",
        f"- formal population: {overall.get('n')}",
        f"- parsed predictions: {overall.get('parsed_n')}",
        f"- parse rate: {overall.get('parse_rate')}",
        f"- exact accuracy: {overall.get('acc_exact')}",
        f"- within-1 accuracy: {overall.get('acc_within_1')}",
        f"- MAE/RMSE/bias: {overall.get('mae')} / {overall.get('rmse')} / {overall.get('bias')}",
        "",
        "## FO1 diagnostics",
        "",
        f"- target phrase supported rate: {diagnostics.get('target_phrase_supported_rate')}",
        f"- proposal count mean/p50/p90: {proposal_counts.get('mean')} / "
        f"{proposal_counts.get('p50')} / {proposal_counts.get('p90')}",
        f"- selected region count mean/p50/p90: {selected_counts.get('mean')} / "
        f"{selected_counts.get('p50')} / {selected_counts.get('p90')}",
        f"- proposal failures: {diagnostics.get('proposal_failure_count')}",
        f"- FO1 parse failures: {diagnostics.get('fo1_parse_failure_count')}",
    ]
    return "\n".join(lines) + "\n"


def evaluate(settings: Mapping[str, Any]) -> dict[str, Path]:
    input_path = Path(settings["input"]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    rows = _select_rows(_read_jsonl(input_path), str(settings["scope"]))
    expected_population = settings.get("expected_population")
    if expected_population is not None and len(rows) != int(expected_population):
        raise ValueError(
            f"{settings['scope']} population mismatch: expected {expected_population}, "
            f"selected {len(rows)} from {input_path}"
        )
    if settings["max_samples"] is not None:
        rows = rows[: int(settings["max_samples"])]
    image_root = Path(settings["image_root"]).resolve() if settings["image_root"] else None
    requests: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for row in rows:
        image, question, reference = _message_parts(row)
        target = extract_count_target_phrase(question)
        phrase = target.phrase or ""
        requests.append(
            {
                "id": str(row["id"]),
                "image": _resolve_image(image, image_root),
                "question": question,
                "target_phrase": phrase,
            }
        )
        prepared.append(
            {
                "source": row,
                "image": image,
                "question": question,
                "reference": reference,
                "target": target,
            }
        )
    started = time.perf_counter()
    responses = _run_worker(requests, settings)
    elapsed = time.perf_counter() - started
    by_id = {str(response.get("id")): response for response in responses}
    predictions: list[dict[str, Any]] = []
    for item in prepared:
        row = item["source"]
        response = dict(
            by_id.get(
                str(row["id"]),
                {
                    "id": row["id"],
                    "status": "failed",
                    "failure_stage": "worker_process",
                    "error": "missing worker response",
                },
            )
        )
        status = str(response.get("status", "failed"))
        count = response.get("fo1_count") if status == "ok" else None
        output = {
            "id": str(row["id"]),
            "task_type": "counting",
            "question": item["question"],
            "reference": item["reference"],
            "prediction": prediction_count_text(count if isinstance(count, int) else None),
            "metadata": dict(row.get("metadata", {}))
            if isinstance(row.get("metadata"), Mapping)
            else {},
            "inference_latency_ms": (
                float(response.get("upn_latency_ms", 0.0))
                + float(response.get("fo1_latency_ms", 0.0))
                if response.get("upn_latency_ms") is not None
                and response.get("fo1_latency_ms") is not None
                else None
            ),
            "prompt_profile": settings["prompt_profile"],
            "target_phrase": item["target"].phrase,
            "target_status": item["target"].status,
            "target_reason": item["target"].reason,
            "worker_status": status,
            "worker_failure_stage": response.get("failure_stage"),
            "worker_error": response.get("error"),
            "proposal_count_raw": response.get("proposal_count_raw"),
            "proposal_count_used": response.get("proposal_count_used"),
            "proposal_boxes": response.get("proposal_boxes", []),
            "proposal_scores": response.get("proposal_scores", []),
            "fo1_raw_output": response.get("fo1_raw_output", ""),
            "fo1_selected_region_indexes": response.get(
                "fo1_selected_region_indexes", response.get("selected_region_indexes", [])
            ),
            "selected_region_indexes": response.get("selected_region_indexes", []),
            "selected_region_boxes": response.get("selected_region_boxes", []),
            "selected_region_scores": response.get("selected_region_scores", []),
            "fo1_textual_count": response.get("fo1_textual_count"),
        }
        predictions.append(output)
    formal = summarize_exact_cardinality_counting(predictions)
    diagnostics = _diagnostics(responses)
    audit_path = Path(settings["audit"]).resolve()
    provenance = {
        "schema_version": "vlm-fo1-evaluation-v1",
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": settings["scope"],
        "input_file": str(input_path),
        "input_sha256": _sha256(input_path),
        "input_rows_selected": len(rows),
        "expected_population": settings.get("expected_population"),
        "prompt_profile": settings["prompt_profile"],
        "backend": settings["backend"],
        "worker_python": str(Path(settings["worker_python"]).resolve()),
        "worker_script": str(Path(settings["worker_script"]).resolve()),
        "cache_dir": os.environ.get("VLM_FO1_CACHE_DIR"),
        "model": str(settings["model"]),
        "upn_checkpoint": str(settings["upn_checkpoint"]),
        "image_root": str(image_root) if image_root else None,
        "proposal_score_threshold": settings["proposal_score_threshold"],
        "proposal_top_k": settings["proposal_top_k"],
        "nms_threshold": settings["nms_threshold"],
        "max_new_tokens": settings["max_new_tokens"],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "formal_counting_protocol": formal["metrics_protocol"],
        "reference_not_sent_to_worker": True,
        "worker_elapsed_seconds": elapsed,
        "source_audit": str(audit_path),
        "source_audit_sha256": _sha256(audit_path) if audit_path.is_file() else None,
    }
    metrics = {
        "schema_version": "vlm-fo1-metrics-v1",
        "metrics_protocol": formal["metrics_protocol"],
        "n": formal["overall"]["n"],
        "parsed_n": formal["overall"]["parsed_n"],
        "parse_rate": formal["overall"]["parse_rate"],
        "acc_exact": formal["overall"]["acc_exact"],
        "acc_within_1": formal["overall"]["acc_within_1"],
        "MAE": formal["overall"]["mae"],
        "RMSE": formal["overall"]["rmse"],
        "bias": formal["overall"]["bias"],
        "overall": formal["overall"],
        "count_bins": formal["count_bins"],
        "formal_counting": formal,
    }
    output_dir = Path(settings["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": output_dir / "predictions.jsonl",
        "metrics": output_dir / "metrics.json",
        "summary": output_dir / "summary.md",
        "provenance": output_dir / "provenance.json",
        "diagnostics": output_dir / "fo1_diagnostics.json",
    }
    outputs["predictions"].write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    outputs["metrics"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs["provenance"].write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs["diagnostics"].write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs["summary"].write_text(
        _summary_markdown(metrics, diagnostics, provenance), encoding="utf-8"
    )
    return outputs


def main() -> int:
    try:
        settings = _settings(parse_args())
        for name, path in evaluate(settings).items():
            print(f"Saved {name}: {path}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"VLM-FO1 evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
