"""统一 Qwen3-VL baseline / INT8 公平 benchmark 入口。"""

# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.quantization.benchmark import run_benchmark
from sat_rs_vlm.quantization.config import load_quantization_config
from sat_rs_vlm.quantization.quantizer import create_backend

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-VL quantization backends.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--backend",
        choices=("baseline", "torch_dynamic_int8", "bnb_int8"),
        default=None,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    forced_backend: str | None = None,
) -> int:
    args = parse_args(argv)
    if forced_backend and args.backend and args.backend != forced_backend:
        raise SystemExit(
            f"Compatibility wrapper requires backend={forced_backend}, got {args.backend}"
        )
    backend_name = forced_backend or args.backend
    overrides = {
        "quantization.backend": backend_name,
        "output.output_dir": args.output_dir,
    }
    try:
        config = load_quantization_config(args.config, overrides=overrides)
        backend = create_backend(config.quantization.backend)
        report = run_benchmark(
            config,
            backend,
            project_root=PROJECT_ROOT,
            skip_baseline=bool(args.skip_baseline),
            dry_run=bool(args.dry_run),
        )
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if bool(report.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
