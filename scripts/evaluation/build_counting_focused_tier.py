"""Build the immutable-source E_COUNT_V1 evaluation tier."""

from __future__ import annotations

import argparse
import json

from sat_rs_vlm.evaluation.counting_focused_tier import (
    DEFAULT_COUNTING_FOCUSED_FILE,
    DEFAULT_COUNTING_FOCUSED_MANIFEST,
    build_counting_focused_tier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default="data/evaluation/tiers/e1_quick.jsonl")
    parser.add_argument("--e2", default="data/evaluation/tiers/e2_standard.jsonl")
    parser.add_argument("--output", default=DEFAULT_COUNTING_FOCUSED_FILE)
    parser.add_argument("--manifest", default=DEFAULT_COUNTING_FOCUSED_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_counting_focused_tier(
        e1_path=args.e1,
        e2_path=args.e2,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
