"""构建 H2 mining candidates；不加载模型、不执行推理。"""

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
from sat_rs_vlm.training.refinement_dataset import (  # noqa: E402
    build_h2_mining_candidates,
    load_protected_e3,
    load_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--manifest-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config)
    h2 = config.h2_refinement
    if not h2.enabled:
        raise ValueError("h2_refinement.enabled must be true")
    train_value = args.train_file or h2.source_training_file
    if train_value is None:
        raise ValueError("H2 source training file is required")
    train_file = resolve_path(train_value, PROJECT_ROOT)
    evaluation_manifest = resolve_path(
        args.evaluation_manifest or h2.protected_evaluation_manifest,
        PROJECT_ROOT,
    )
    output_file = resolve_path(args.output_file or h2.mining_candidates_file, PROJECT_ROOT)
    manifest_file = resolve_path(
        args.manifest_file or h2.mining_candidates_manifest,
        PROJECT_ROOT,
    )
    manifest = build_h2_mining_candidates(
        load_rows(train_file),
        load_protected_e3(evaluation_manifest),
        h2,
        output_file=output_file,
        manifest_file=manifest_file,
        source_training_file=train_file,
        bbox_small_max=config.hard_adaptation.bbox_area_thresholds.small_max,
        bbox_medium_max=config.hard_adaptation.bbox_area_thresholds.medium_max,
    )
    print(
        json.dumps(
            {
                "output_file": str(output_file),
                "manifest_file": str(manifest_file),
                "sample_count": manifest["actual_samples"],
                "sha256": manifest["output"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
