"""从层敏感度报告生成混合 bitsandbytes INT8 benchmark 配置。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.quantization.mixed import build_mixed_precision_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--sensitivity-report", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument(
        "--keep-top-groups",
        type=int,
        default=0,
        help=(
            "Also preserve this many highest-scoring groups; useful when no group "
            "crosses threshold."
        ),
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return dict(loaded)


def main() -> int:
    args = parse_args()
    base_config = _read_yaml(args.base_config)
    sensitivity_report = json.loads(args.sensitivity_report.read_text(encoding="utf-8"))
    if not isinstance(sensitivity_report, dict):
        raise ValueError("Expected a JSON object sensitivity report")
    mixed_config, summary = build_mixed_precision_config(
        base_config,
        sensitivity_report,
        keep_top_groups=args.keep_top_groups,
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(mixed_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_config": str(args.output_config),
                "sensitivity_report": str(args.sensitivity_report),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
