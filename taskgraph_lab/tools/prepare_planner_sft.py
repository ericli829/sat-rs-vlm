from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskgraph_lab.training.planner_dataset import PlannerSFTDataset

LAB_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic TaskGraph Planner SFT data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=LAB_ROOT / "prompts" / "planner_student_system_prompt.txt",
    )
    parser.add_argument("--target-format", choices=("dsl", "json"), default="dsl")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = PlannerSFTDataset(
        args.input,
        system_prompt=args.system_prompt,
        target_format=args.target_format,
    )
    manifest = dataset.write_splits(
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
