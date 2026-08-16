"""Use a local small language model to classify LEVIR-CC caption semantics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.change_judge import (  # noqa: E402
    HuggingFaceQwenJudge,
    run_local_change_judge,
)
from sat_rs_vlm.evaluation.records import EvaluationError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--routing", choices=("all", "cascade"), default="cascade")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the model loader to access the network when local files are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        backend = HuggingFaceQwenJudge(
            args.model,
            batch_size=args.batch_size,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            local_files_only=not args.allow_download,
            model_revision=args.model_revision,
        )
        outputs = run_local_change_judge(
            args.predictions,
            args.output_dir,
            backend,
            routing=args.routing,
            strict=args.strict,
        )
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Local judge failed: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
