from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from taskgraph_lab.taskgraph.canonicalize import stable_json_dumps


def _records(paths: list[Path]) -> Any:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _review_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    for record in _records([path]):
        if record.get("status") == "reviewed":
            result[str(record["sample_id"])] = str(record["review"]["verdict"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export validated TaskGraphs for Planner SFT")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("target", "messages"), default="target")
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "prompts/system_prompt.txt",
    )
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--allow-non-minimal", action="store_true")
    args = parser.parse_args()
    reviews = _review_map(args.reviews)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in _records(args.input):
            verdict = reviews.get(str(record["sample_id"]))
            allowed = {None, "VALID"}
            if args.allow_non_minimal:
                allowed.add("VALID_BUT_NON_MINIMAL")
            if verdict not in allowed:
                continue
            base = {
                "sample_id": record["sample_id"],
                "input": record["input"],
                "target": record["target"],
                "metadata": record.get("metadata", {}),
            }
            if "planner_dsl" in record:
                base["planner_dsl"] = record["planner_dsl"]
            if args.format == "messages":
                payload = {
                    "sample_id": record["sample_id"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(record["input"], ensure_ascii=False),
                        },
                        {"role": "assistant", "content": stable_json_dumps(record["target"])},
                    ],
                    "metadata": record.get("metadata", {}),
                }
            else:
                payload = base
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "records": count, "format": args.format},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
