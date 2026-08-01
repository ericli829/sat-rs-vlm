"""按 task_type 生成分层或显式配额 JSONL。"""

from __future__ import annotations

import argparse
from pathlib import Path

from sat_rs_vlm.data.sampling import allocate_quotas, group_by_task, sample_by_task
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def parse_quotas(text: str) -> dict[str, int]:
    """解析 `task=count` 逗号列表。"""

    quotas: dict[str, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid quota item '{item}', expected task=count")
        task, count = item.split("=", 1)
        quotas[task.strip()] = int(count.strip())
    if not quotas:
        raise ValueError("Quota list is empty")
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample JSONL by task type.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--total", type=int)
    group.add_argument("--per-task", type=int)
    group.add_argument("--quotas")
    parser.add_argument("--tasks", default=None, help="Optional comma-separated allowlist.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-replacement", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"Input JSONL does not exist: {source}")
    rows = list(read_jsonl(source))
    if args.tasks:
        allowed = {item.strip() for item in args.tasks.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("task_type", "unknown")) in allowed]
    grouped = group_by_task(rows)
    explicit = parse_quotas(args.quotas) if args.quotas else None
    quotas = allocate_quotas(
        grouped,
        total=args.total,
        per_task=args.per_task,
        explicit=explicit,
        with_replacement=bool(args.with_replacement),
    )
    selected, stats = sample_by_task(
        rows,
        quotas,
        seed=args.seed,
        with_replacement=bool(args.with_replacement),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    print(f"Wrote {len(selected)} samples to {output}; task_counts={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
