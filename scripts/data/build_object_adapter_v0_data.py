"""Build and audit the frozen RS Object Adapter v0 training assets.

This entry point intentionally performs no model loading.  It reads the formal
VRSBench-only source and the protected evaluation population, removes image-level
overlap, and writes the five immutable assets required by the experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.configuration.environment import expand_environment  # noqa: E402
from sat_rs_vlm.data.object_adapter_v0 import (  # noqa: E402
    DataAuditBlocked,
    build_object_adapter_dataset,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Object Adapter config must be a mapping: {path}")
    # The builder does not need DATA_ROOT because it never opens images.  Keep
    # that placeholder unresolved here so a local data-audit attempt can reach
    # the formal source-file check and report the real missing asset.
    expanded = expand_environment(payload, environ=os.environ, allow_unresolved=True)
    return dict(expanded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/rs_object_adapter_v0_4090.yaml"),
    )
    parser.add_argument("--allow-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = _project_path(args.config)
    config = _load_config(config_path)
    data = dict(config.get("data", {}))
    train_source = _project_path(str(data["train_source"]))
    protected_eval_source = _project_path(str(data["protected_eval_source"]))
    tier_manifest = data.get("evaluation_tier_manifest")
    missing_assets = [
        str(path)
        for path in (train_source, protected_eval_source)
        if not path.is_file()
    ]
    if tier_manifest:
        tier_path = _project_path(str(tier_manifest))
        if not tier_path.is_file():
            missing_assets.append(str(tier_path))
    if missing_assets:
        raise FileNotFoundError(
            "Formal RS Object Adapter v0 assets are missing; refusing to substitute legacy "
            "data: " + "; ".join(missing_assets)
        )
    output_dir = _project_path(str(data.get("output_dir", "data/processed/rs_object_adapter_v0")))
    try:
        manifest = build_object_adapter_dataset(
            train_source,
            protected_eval_source,
            output_dir=output_dir,
            seed=int(config.get("experiment", {}).get("seed", 42)),
            val_fraction=float(data.get("internal_val_fraction", 0.05)),
            dedup_iou=float(data.get("detection_dedup_iou", 0.95)),
            enforce_blockers=not args.allow_blocked,
        )
    except DataAuditBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Object Adapter v0 data assets written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
