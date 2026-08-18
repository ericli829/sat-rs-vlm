"""构建 Qwen3-VL-4B last-2 visual probe 的固定训练 JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.training.config import load_training_config  # noqa: E402
from sat_rs_vlm.training.vit_probe import build_probe_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-train-file", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config)
    probe = config.vit_probe
    source_files = args.source_train_file or probe.source_train_files
    output_dir = args.output_dir or Path(probe.output_dir)
    manifest = build_probe_dataset(
        source_files,
        output_dir=output_dir,
        protected_evaluation_manifest=probe.protected_evaluation_manifest,
        target_samples=args.target_samples or probe.target_samples,
        source_targets=probe.source_targets,
        task_targets=probe.task_targets,
        seed=args.seed if args.seed is not None else probe.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

