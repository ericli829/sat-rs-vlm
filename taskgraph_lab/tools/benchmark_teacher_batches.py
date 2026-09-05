from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from taskgraph_lab.generation.batch_generation import generate_teacher_batch
from taskgraph_lab.generation.generate import (
    RateLimiter,
    RuntimeSettings,
    append_jsonl,
    iter_samples,
)
from taskgraph_lab.generation.provider import provider_from_config

LAB_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("batch benchmark config must be a YAML mapping")
    return payload


def _write_run_outputs(output_dir: Path, size: int, result: Any) -> None:
    run_dir = output_dir / f"batch_{size}"
    for outcome in result.outcomes:
        append_jsonl(run_dir / "raw.jsonl", outcome.raw)
        if outcome.destination and outcome.record is not None:
            append_jsonl(run_dir / f"{outcome.destination}.jsonl", outcome.record)
    calls_path = run_dir / "calls.json"
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.write_text(
        json.dumps(
            [
                {
                    "call_id": call.call_id,
                    "kind": call.kind,
                    "batch_id": call.batch_id,
                    "sample_ids": call.sample_ids,
                    "trace": call.trace,
                    "transport": call.transport,
                    "error": call.error,
                }
                for call in result.calls
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row["batch_size"] == 1), None)
    compared: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        if baseline:
            base_calls = float(baseline["calls_per_sample"] or 0)
            base_prompt = float(baseline["prompt_tokens_per_sample"] or 0)
            enriched["api_call_saving_vs_batch_1"] = (
                1.0 - float(row["calls_per_sample"]) / base_calls if base_calls else None
            )
            enriched["prompt_tokens_per_sample_drop_vs_batch_1"] = (
                1.0 - float(row["prompt_tokens_per_sample"]) / base_prompt if base_prompt else None
            )
        compared.append(enriched)
    return compared


def _recommend(rows: list[dict[str, Any]]) -> int | None:
    eligible = [
        row
        for row in rows
        if row["transport_parse_failures"] == 0
        and row["api_failed"] == 0
        and row["rejected"] == min(item["rejected"] for item in rows)
    ]
    if not eligible:
        return None
    return int(
        max(
            eligible,
            key=lambda row: (
                row["accepted_per_api_call"],
                -row["prompt_tokens_per_sample"],
                -row["batch_size"],
            ),
        )["batch_size"]
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TaskGraph batch Teacher benchmark",
        "",
        "All runs use the same ordered sample set and DeepSeek thinking-low mode.",
        "",
        "| batch | samples | calls | transport failures | valid | repaired | rejected | "
        "prompt/sample | completion/sample | calls/sample | accepted/call |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["runs"]:
        lines.append(
            "| {batch_size} | {sample_count} | {api_calls} | {transport_parse_failures} | "
            "{initial_valid} | {repaired} | {rejected} | {prompt_tokens_per_sample:.2f} | "
            "{completion_tokens_per_sample:.2f} | {calls_per_sample:.3f} | "
            "{accepted_per_api_call:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Recommended production batch size: {report['recommended_batch_size']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    *,
    input_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    system_prompt: str,
    batch_contract: str,
    batch_sizes: list[int],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"benchmark output directory must be empty: {output_dir}")
    samples = list(iter_samples(input_path))
    settings = RuntimeSettings.from_mapping(dict(config.get("runtime") or {}))
    provider = provider_from_config(dict(config.get("provider") or {}))
    limiter = RateLimiter(settings.requests_per_minute)
    batch_config = dict(config.get("batch") or {})
    rows: list[dict[str, Any]] = []
    for size in batch_sizes:
        print(
            json.dumps(
                {
                    "event": "batch_size_started",
                    "batch_size": size,
                    "sample_count": len(samples),
                }
            ),
            flush=True,
        )
        result = generate_teacher_batch(
            samples,
            provider=provider,
            limiter=limiter,
            settings=settings,
            system_prompt=system_prompt,
            batch_transport_contract=batch_contract,
            batch_size=size,
            teacher_batch_max_input_tokens=int(
                batch_config.get("teacher_batch_max_input_tokens", 24000)
            ),
            teacher_batch_max_samples=int(batch_config.get("teacher_batch_max_samples", 8)),
            max_transport_retries=int(batch_config.get("max_transport_retries", 1)),
            progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
        )
        _write_run_outputs(output_dir, size, result)
        metrics = result.metrics()
        rows.append(metrics)
        print(
            json.dumps(
                {"event": "batch_size_completed", "batch_size": size, "metrics": metrics},
                ensure_ascii=False,
            ),
            flush=True,
        )
    compared = _comparison(rows)
    report = {
        "input": str(input_path.resolve()),
        "sample_ids": [sample.sample_id for sample in samples],
        "thinking_mode": "low",
        "runs": compared,
        "recommended_batch_size": _recommend(compared),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "batch_benchmark.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TaskGraph batch Teacher transport")
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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    report = run_benchmark(
        input_path=args.input,
        config=_load_config(args.config),
        output_dir=args.output_dir,
        system_prompt=args.system_prompt.read_text(encoding="utf-8"),
        batch_contract=args.batch_contract.read_text(encoding="utf-8"),
        batch_sizes=args.batch_sizes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
