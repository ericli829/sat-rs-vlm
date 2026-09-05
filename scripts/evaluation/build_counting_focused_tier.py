"""Build E_COUNT_V2 from formal unified-v2 tiers (or legacy V1 explicitly)."""

from __future__ import annotations

import argparse
import json

from sat_rs_vlm.evaluation.counting_focused_tier import (
    DEFAULT_COUNTING_FOCUSED_FILE,
    DEFAULT_COUNTING_FOCUSED_MANIFEST,
    DEFAULT_COUNTING_FOCUSED_V2_FILE,
    DEFAULT_COUNTING_FOCUSED_V2_MANIFEST,
    DEFAULT_UNIFIED_E1_FILE,
    DEFAULT_UNIFIED_E2_FILE,
    DEFAULT_UNIFIED_MANIFEST,
    build_counting_focused_tier,
    build_counting_focused_tier_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-v1", action="store_true")
    parser.add_argument("--e1")
    parser.add_argument("--e2")
    parser.add_argument("--source-manifest", default=DEFAULT_UNIFIED_MANIFEST)
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.legacy_v1:
        manifest = build_counting_focused_tier(
            e1_path=args.e1 or "data/evaluation/tiers/e1_quick.jsonl",
            e2_path=args.e2 or "data/evaluation/tiers/e2_standard.jsonl",
            output_path=args.output or DEFAULT_COUNTING_FOCUSED_FILE,
            manifest_path=args.manifest or DEFAULT_COUNTING_FOCUSED_MANIFEST,
        )
    else:
        manifest = build_counting_focused_tier_v2(
            e1_path=args.e1 or DEFAULT_UNIFIED_E1_FILE,
            e2_path=args.e2 or DEFAULT_UNIFIED_E2_FILE,
            source_manifest_path=args.source_manifest,
            output_path=args.output or DEFAULT_COUNTING_FOCUSED_V2_FILE,
            manifest_path=args.manifest or DEFAULT_COUNTING_FOCUSED_V2_MANIFEST,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
