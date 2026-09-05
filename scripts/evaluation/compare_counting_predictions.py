#!/usr/bin/env python3
"""Compare aligned counting prediction JSONL files without router search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.ensemble import (  # noqa: E402
    majority_vote_counting,
    median_vote_counting,
    pairwise_counting_comparison,
)


def _read(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"prediction row must be an object: {path}")
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions", nargs="+", type=Path, help="At least two prediction JSONL files."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.predictions) < 2:
        parser.error("at least two prediction files are required")
    try:
        rows = [_read(path) for path in args.predictions]
        payload: dict[str, object] = {
            "candidate_files": [str(path.resolve()) for path in args.predictions],
            "pairwise": pairwise_counting_comparison(rows[0], rows[1]),
            "majority": majority_vote_counting(rows),
            "median": median_vote_counting(rows),
            "router_threshold_search": {"performed": False, "development_only": False},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(args.output)
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
