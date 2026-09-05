"""Real-data throughput benchmark with one 4B CUDA context per setting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

from sat_rs_vlm.utils.jsonl import read_jsonl

SETTINGS = ((4, 4), (8, 2), (16, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=64, choices=(64, 128))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="outputs/rs_merger_throughput")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sample_ids = [
        str(row.get("id", ""))
        for row in list(read_jsonl(source["data"]["train_file"]))[: args.samples]
    ]
    results = []
    for batch_size, accumulation in SETTINGS:
        config = json.loads(json.dumps(source))
        name = f"b{batch_size}_a{accumulation}"
        config["experiment"] = f"throughput_{name}"
        config["training"]["per_device_train_batch_size"] = batch_size
        config["training"]["gradient_accumulation_steps"] = accumulation
        config_path = output / f"{name}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        run_root = output / name
        max_steps = math.ceil(args.samples / (batch_size * accumulation))
        command = [
            args.python,
            "scripts/training/train_rs_merger_expert.py",
            "--config",
            str(config_path),
            "--max-train-samples",
            str(args.samples),
            "--max-steps",
            str(max_steps),
            "--output-root",
            str(run_root),
        ]
        log_path = output / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            return_code = process.wait()
        record = {
            "setting": name,
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch_size * accumulation,
            "max_steps": max_steps,
            "pid": process.pid,
            "return_code": return_code,
            "log": str(log_path),
            "cuda_context_boundary": "child exited before next setting",
        }
        summaries = sorted(run_root.rglob("training_summary.json"))
        if summaries:
            record["training_summary"] = json.loads(summaries[-1].read_text(encoding="utf-8"))
        results.append(record)
        (output / "throughput_report.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "real_sample_count": args.samples,
                    "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
                    "settings": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
