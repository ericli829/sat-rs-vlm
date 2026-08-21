"""Train RS Object Adapter v0 on a frozen Qwen3-VL-4B R1 visual tower.

The script is deliberately a thin entry point.  Data construction, visual hook
extraction, Hungarian matching, loss computation, checkpoint saving, and audit
logic live in :mod:`sat_rs_vlm.training.object_adapter_v0`.
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
from sat_rs_vlm.data.object_adapter_v0 import DataAuditBlocked  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Object Adapter config must be a mapping: {path}")
    expanded = expand_environment(payload, environ=os.environ, allow_unresolved=False)
    return dict(expanded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/rs_object_adapter_v0_4090.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override model.checkpoint_dir with the existing Qwen3-VL-4B R1 adapter.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-groups", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--max-val-groups",
        type=int,
        default=None,
        help="Limit internal validation batches; intended for real-model smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = _project_path(args.config)
    config = _load_config(config_path)
    if args.checkpoint_dir is not None:
        config.setdefault("model", {})["checkpoint_dir"] = str(_project_path(args.checkpoint_dir))
    if args.output_dir is not None:
        config.setdefault("training", {})["output_dir"] = str(_project_path(args.output_dir))
    try:
        from sat_rs_vlm.training.object_adapter_v0 import run_object_adapter_training

        result = run_object_adapter_training(
            config,
            project_root=PROJECT_ROOT,
            max_train_groups=args.max_train_groups,
            max_steps=args.max_steps,
            max_val_groups=args.max_val_groups,
            dry_run=args.dry_run,
        )
    except (DataAuditBlocked, FileNotFoundError, ImportError, ValueError, OSError) as exc:
        print(f"Object Adapter v0 training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
