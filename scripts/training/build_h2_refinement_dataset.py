"""从 Evaluation v1.5 mining 结果构建 H2 final refinement dataset。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.training.config import load_training_config, resolve_path  # noqa: E402
from sat_rs_vlm.training.hard_example_mining import (  # noqa: E402
    resolve_evaluated_predictions,
)
from sat_rs_vlm.training.refinement_dataset import (  # noqa: E402
    build_h2_refinement_dataset,
    load_protected_e3,
    load_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--evaluated-predictions", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-checkpoint", type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config)
    h2 = config.h2_refinement
    if not h2.enabled:
        raise ValueError("h2_refinement.enabled must be true")
    if args.source_checkpoint and args.source_checkpoint != h2.source_checkpoint:
        raise ValueError(
            "--source-checkpoint must match the Replay generalist checkpoint in config"
        )
    train_value = args.train_file or h2.source_training_file
    predictions_value = args.evaluated_predictions or h2.evaluated_predictions_file
    if train_value is None or predictions_value is None:
        raise ValueError("H2 source training file and evaluated predictions are required")
    train_file = resolve_path(train_value, PROJECT_ROOT)
    candidates_file = resolve_path(args.candidates or h2.mining_candidates_file, PROJECT_ROOT)
    predictions_file = resolve_evaluated_predictions(
        resolve_path(predictions_value, PROJECT_ROOT)
    )
    evaluation_manifest = resolve_path(
        args.evaluation_manifest or h2.protected_evaluation_manifest,
        PROJECT_ROOT,
    )
    output_dir = resolve_path(args.output_dir or h2.output_dir, PROJECT_ROOT)
    manifest = build_h2_refinement_dataset(
        load_rows(train_file),
        load_rows(candidates_file),
        load_rows(predictions_file),
        load_protected_e3(evaluation_manifest),
        h2,
        config.hard_adaptation,
        source_training_file=train_file,
        mining_candidates_file=candidates_file,
        prediction_source=predictions_file,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": manifest["target_samples"],
                "h2_train_sha256": manifest["output_sha256"]["h2_train"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
