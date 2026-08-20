"""Build a strict balanced probe from the Stage-A v2 canonical population."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.probe_sampling import build_balanced_probe_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(expand_environment(payload, environ=os.environ, allow_unresolved=False))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = build_balanced_probe_dataset(
        config["population_manifest"],
        targets=dict(config["targets"]),
        output_dir=args.output_dir or config["output_dir"],
        seed=int(args.seed if args.seed is not None else config.get("seed", 42)),
        quota_shortfall_policy=str(config.get("quota_shortfall_policy", "error")),
        duplicate_policy=str(config.get("duplicate_policy", "error")),
        total_samples=int(config["total_samples"]),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
