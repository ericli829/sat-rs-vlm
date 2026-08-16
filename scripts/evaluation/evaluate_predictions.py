"""对已有 predictions.jsonl 执行 v1.6 多任务离线评测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.records import EvaluationError  # noqa: E402
from sat_rs_vlm.evaluation.runner import run_evaluation  # noqa: E402

DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "eval" / "evaluation_contract_v1.7.yaml"
DEFAULT_SEMANTIC_CONTRACT = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "semantic_contract.json"
)
DEFAULT_SEMANTIC_ONTOLOGY = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "remote_sensing_ontology.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--semantic-contract", type=Path, default=DEFAULT_SEMANTIC_CONTRACT)
    parser.add_argument("--semantic-ontology", type=Path, default=DEFAULT_SEMANTIC_ONTOLOGY)
    parser.add_argument(
        "--latency-semantics",
        choices=("unresolved", "single_sample", "batch_amortized_per_sample"),
        default="unresolved",
    )
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--group-by-task", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--protect-repository",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refuse output paths inside this repository (disabled for normal integrated use).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_evaluation(
            args.predictions,
            args.output_dir,
            contract_path=args.contract,
            manifest_path=args.manifest,
            strict=args.strict,
            protected_repository=PROJECT_ROOT if args.protect_repository else None,
            semantic_enabled=args.semantic,
            semantic_contract_path=args.semantic_contract,
            semantic_ontology_path=args.semantic_ontology,
            latency_semantics=args.latency_semantics,
            eval_batch_size=args.eval_batch_size,
            group_by_task=args.group_by_task,
        )
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
