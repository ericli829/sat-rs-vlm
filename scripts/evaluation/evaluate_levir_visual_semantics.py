"""Evaluate LEVIR captions against image-audited visual-semantic gold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.visual_semantic_runner import (  # noqa: E402
    evaluate_visual_semantics,
    read_prediction_outputs,
    write_visual_semantic_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-profile", required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete-historical-manifest",
        action="store_true",
    )
    parser.add_argument(
        "--verify-image-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    result = evaluate_visual_semantics(
        read_prediction_outputs(args.predictions),
        gold_csv=args.gold_csv,
        predictions_path=args.predictions,
        generation_manifest_path=args.generation_manifest,
        prompt_profile=args.prompt_profile,
        allow_incomplete_historical_manifest=args.allow_incomplete_historical_manifest,
        verify_image_paths=args.verify_image_paths,
    )
    outputs = write_visual_semantic_evaluation(result, output_dir)
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
