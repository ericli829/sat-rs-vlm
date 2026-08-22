"""Build the leakage-protected Counting Expert v1 population."""

from __future__ import annotations

import argparse
import json

from sat_rs_vlm.data.rs_merger_expert import build_counting_expert_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", default="data/processed/rs_count_merger_v1")
    parser.add_argument(
        "--protected-tier",
        action="append",
        dest="protected_tiers",
        default=None,
        help="Repeat for E1/E2/E3; defaults to the repository frozen tiers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tiers = args.protected_tiers or [
        "data/evaluation/tiers/e1_quick.jsonl",
        "data/evaluation/tiers/e2_standard.jsonl",
        "data/evaluation/tiers/e3_full.jsonl",
    ]
    result = build_counting_expert_data(
        args.source_train,
        args.output_dir,
        protected_tiers=tiers,
        source_manifest=args.source_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
