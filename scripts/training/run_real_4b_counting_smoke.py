"""End-to-end real 4B smoke for C2, C3, and one capacity route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

CONFIGS = (
    "configs/experiments/rs_count_merger_c2_count_4e.yaml",
    "configs/experiments/rs_count_merger_c3_count_4e.yaml",
    "configs/experiments/rs_count_merger_c4_wide_count_4e.yaml",
)
FORMAL_SIDECAR_SHA256 = "67c2b33d255492080166efc767d1fceb46e007184b162f481f274d5327b989ae"
LOCAL_LOW_MEMORY_MARKERS = (
    "out of memory",
    "need an `offload_dir` to dispatch this model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--r1-checkpoint", required=True)
    parser.add_argument("--visual-sidecar", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--eval-image-root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--steps", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output-dir", default="outputs/local_real_4b_counting_smoke")
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIGS,
        help="Run only selected smoke configs; repeat for multiple. Defaults to the full matrix.",
    )
    parser.add_argument(
        "--low-memory-evidence-log",
        help="Prior BF16 log containing an accepted low-memory marker; enables direct real NF4.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        return process.wait()


def is_local_low_memory_failure(log_text: str) -> bool:
    """Recognize only failures that justify the real NF4 smoke fallback."""

    normalized = log_text.lower()
    return any(marker in normalized for marker in LOCAL_LOW_MEMORY_MARKERS)


def build_real4bit_fallback_config(source: dict[str, object]) -> dict[str, object]:
    """Build the local low-memory config without changing the formal BF16 source."""

    fallback = deepcopy(source)
    model = dict(fallback["model"])
    model["load_in_4bit"] = True
    # The R1 adapter remains active and frozen. Quantized PEFT merge is not a
    # parity-preserving operation, while additive is the proven C2/C3 runtime.
    model["r1_integration"] = "additive"
    # Keep formal visual-sidecar tensors shape-compatible and retain the output
    # embedding/head in its normal dtype; quantize the language transformer.
    model["quantization_skip_modules"] = ["model.visual", "lm_head"]
    fallback["model"] = model
    return fallback


def validated_low_memory_reason(path: str | None) -> str | None:
    if path is None:
        return None
    evidence = Path(path)
    if not evidence.is_file():
        raise FileNotFoundError(f"Low-memory evidence log does not exist: {evidence}")
    log_text = evidence.read_text(encoding="utf-8", errors="replace").lower()
    if not is_local_low_memory_failure(log_text):
        raise ValueError(f"Low-memory evidence contains no accepted marker: {evidence}")
    marker = next(marker for marker in LOCAL_LOW_MEMORY_MARKERS if marker in log_text)
    return f"validated_evidence:{evidence.resolve()}:{marker}"


def main() -> int:
    args = parse_args()
    base = Path(args.base_model)
    r1 = Path(args.r1_checkpoint)
    sidecar = Path(args.visual_sidecar)
    known_low_memory_reason = validated_low_memory_reason(args.low_memory_evidence_log)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    required = [base, r1 / "adapter_model.safetensors", r1 / "strategy_manifest.json", sidecar]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        (output / "real_4b_smoke_report.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "blocked_missing_exact_assets",
                    "missing": missing,
                    "mock_or_substitute_used": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2
    actual_sidecar_sha = _sha256(sidecar)
    if actual_sidecar_sha != FORMAL_SIDECAR_SHA256:
        raise ValueError(
            "Formal R1 visual sidecar SHA256 mismatch: "
            f"expected={FORMAL_SIDECAR_SHA256}, actual={actual_sidecar_sha}"
        )
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_ROOT": str(base.parent),
            "R1_CHECKPOINT": str(r1),
            "R1_VISUAL_SIDECAR": str(sidecar),
            "DATA_ROOT": str(Path(args.image_root)),
            "EVAL_DATA_ROOT": str(
                Path(args.eval_image_root) if args.eval_image_root else Path(args.image_root).parent
            ),
            "OUTPUT_ROOT": str(output),
            "SOURCE_ARCHITECTURE_AUDIT": str(
                output / "runtime_source_audit" / "source_architecture_audit.json"
            ),
        }
    )
    runtime_audit_dir = output / "runtime_source_audit"
    audit_code = _run(
        [
            args.python,
            "scripts/training/audit_rs_merger_architecture.py",
            "--base-model",
            str(base),
            "--r1-checkpoint",
            str(r1),
            "--visual-sidecar",
            str(sidecar),
            "--output-dir",
            str(runtime_audit_dir),
        ],
        output / "source_architecture_audit.log",
        environment,
    )
    if audit_code != 0:
        return audit_code
    results = []
    for source_config in tuple(args.config or CONFIGS):
        name = Path(source_config).stem
        variant_root = output / name
        train_command = [
            args.python,
            "scripts/training/train_rs_merger_expert.py",
            "--config",
            source_config,
            "--max-train-samples",
            "1",
            "--max-steps",
            str(args.steps),
            "--max-eval-samples",
            "2",
            "--output-root",
            str(variant_root),
        ]
        train_log = output / f"{name}_bf16_train.log"
        precision_status = "LOCAL_REAL4B_BF16_PASS"
        precision_fallback_reason = known_low_memory_reason
        used_config = source_config
        train_code = -1
        if known_low_memory_reason is None:
            train_code = _run(train_command, train_log, environment)
        if known_low_memory_reason is not None or train_code != 0:
            log_text = ""
            if known_low_memory_reason is None:
                log_text = train_log.read_text(encoding="utf-8", errors="replace").lower()
                if not is_local_low_memory_failure(log_text):
                    return train_code
                precision_fallback_reason = next(
                    marker for marker in LOCAL_LOW_MEMORY_MARKERS if marker in log_text
                )
            fallback = build_real4bit_fallback_config(
                yaml.safe_load(Path(source_config).read_text(encoding="utf-8"))
            )
            fallback_path = output / f"{name}_real4bit.yaml"
            fallback_path.write_text(
                yaml.safe_dump(fallback, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            used_config = str(fallback_path)
            train_command[train_command.index(source_config)] = used_config
            train_log = output / f"{name}_real4bit_train.log"
            train_code = _run(train_command, train_log, environment)
            precision_status = "LOCAL_REAL4B_4BIT_PASS"
            if train_code != 0:
                return train_code
        checkpoints = sorted(variant_root.rglob("checkpoint/expert_manifest.json"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one smoke checkpoint for {name}, found {checkpoints}")
        checkpoint = checkpoints[0].parent
        training_summary = json.loads(
            (checkpoint / "training_summary.json").read_text(encoding="utf-8")
        )
        if int(training_summary.get("optimizer_steps", -1)) != args.steps:
            raise AssertionError(
                f"{name} optimizer-step smoke mismatch: {training_summary}"
            )
        eval_root = output / f"{name}_eval"
        eval_command = [
            args.python,
            "scripts/evaluation/evaluate_rs_merger_expert.py",
            "--base-model",
            str(base),
            "--r1-checkpoint",
            str(r1),
            "--visual-sidecar",
            str(sidecar),
            "--expert-checkpoint",
            str(checkpoint),
            "--architecture-audit",
            str(runtime_audit_dir / "source_architecture_audit.json"),
            "--tier-file",
            "data/evaluation/tiers_v2/e_count_v2.jsonl",
            "--tier-manifest",
            "data/evaluation/tiers_v2/e_count_v2_manifest.json",
            "--image-root",
            environment["EVAL_DATA_ROOT"],
            "--output-root",
            str(eval_root),
            "--experiment-matrix",
            str(output / "experiment_matrix.md"),
            "--max-eval-samples",
            "2",
            "--max-new-tokens",
            "8",
            "--eval-batch-size",
            "2",
            "--verify-batch1-parity",
        ]
        eval_code = _run(eval_command, output / f"{name}_eval.log", environment)
        if eval_code != 0:
            return eval_code
        results.append(
            {
                "variant": name,
                "precision_status": precision_status,
                "precision_fallback_reason": precision_fallback_reason,
                "config": used_config,
                "checkpoint": str(checkpoint),
                "optimizer_steps": training_summary["optimizer_steps"],
                "trainable_params": training_summary["trainable_params"],
                "memory": training_summary["memory"],
                "save_reload_generate_parser_metric": "pass",
                "train_child_exited_before_next_variant": True,
            }
        )
        (output / "real_4b_smoke_report.json").write_text(
            json.dumps({"schema_version": "1.0", "variants": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
