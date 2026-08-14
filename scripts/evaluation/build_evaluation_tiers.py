"""Build deterministic, nested E1/E2/E3 evaluation assets from validation splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.config import load_evaluation_tier_config  # noqa: E402
from sat_rs_vlm.evaluation.tiers import (  # noqa: E402
    EvaluationTierError,
    build_evaluation_tiers,
)


def parse_args() -> argparse.Namespace:
    """Parse the single YAML entry point used locally and on AutoDL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/evaluation_tiers.yaml"),
        help="Tier builder YAML. Paths may use the project's environment resolver.",
    )
    return parser.parse_args()


def main() -> int:
    """Load typed configuration, generate all assets, and print audit essentials."""

    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    try:
        config = load_evaluation_tier_config(config_path)
        manifest = build_evaluation_tiers(config, project_root=PROJECT_ROOT)
    except (EvaluationTierError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Evaluation tier build failed: {exc}", file=sys.stderr)
        return 2
    print(f"Population: {manifest['population_sample_count']}")
    for name in ("E1", "E2", "E3"):
        tier = manifest["tiers"][name]
        print(
            f"{name}: samples={tier['sample_count']} sha256={tier['sha256']} "
            f"path={tier['path']}"
        )
    invariants = manifest["invariants"]
    print(f"E1 subset of E2: {invariants['E1_subset_of_E2']}")
    print(f"E2 subset of E3: {invariants['E2_subset_of_E3']}")
    print(f"Train/eval disjoint: {invariants['train_eval_disjoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
