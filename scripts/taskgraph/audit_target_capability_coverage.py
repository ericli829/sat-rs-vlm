"""Audit target capability coverage in local MME, XLRS, and planner JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.taskgraph.capability_audit import (
    build_target_capability_coverage,
    write_target_capability_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("configs/eval/semantic/remote_sensing_ontology.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/taskgraph/target_capability_coverage.json"),
    )
    args = parser.parse_args()
    report = build_target_capability_coverage(args.input, ontology_path=args.ontology)
    output = write_target_capability_coverage(report, args.output)
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
