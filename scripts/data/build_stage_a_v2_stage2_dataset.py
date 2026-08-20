"""Build the frozen VRSBench + LEVIR-CC Stage-A v2 R1 exposure dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.data.stage_a_v2 import build_stage2_vrs_levir_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vrs-per-levir", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_stage2_vrs_levir_dataset(
        Path(args.population_manifest),
        output_file=Path(args.output_file),
        manifest_file=Path(args.manifest_file),
        seed=args.seed,
        vrs_per_levir=args.vrs_per_levir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
