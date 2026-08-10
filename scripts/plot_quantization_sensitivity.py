"""从量化敏感度 JSON 报告生成静态图表。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sat_rs_vlm.quantization.config import load_quantization_config
from sat_rs_vlm.quantization.sensitivity import plot_sensitivity_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_quantization_config(args.config)
    sensitivity_dir = _project_path(Path(config.output.output_dir) / "sensitivity")
    report_path = args.report or sensitivity_dir / "sensitivity_report.json"
    output_dir = args.output_dir or sensitivity_dir / "figures"
    report = json.loads(_project_path(report_path).read_text(encoding="utf-8"))
    generated = plot_sensitivity_report(report, _project_path(output_dir))
    print(json.dumps({"generated": [str(path) for path in generated]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
