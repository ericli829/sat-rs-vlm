"""把现有 JSONL 转换为统一结构化 prompt，并报告 unresolved counting。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.prompt_templates import strengthen_answer, strengthen_instruction
from sat_rs_vlm.data.task_protocol import counting_json
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite structured prompts in a JSONL copy.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rewrite_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """重写一行；counting 无法转换时降级为 VQA 并返回 unresolved=True。"""

    updated = dict(row)
    task = str(updated.get("task_type", "unknown"))
    unresolved = False
    answer = updated.get("answer")
    if answer is None and isinstance(updated.get("messages"), list):
        for message in updated["messages"]:
            if message.get("role") == "assistant":
                answer = message.get("content", "")
                break
    if task == "counting" and counting_json(answer) is None:
        unresolved = True
        task = "vqa"
        updated["task_type"] = task
        metadata = dict(updated.get("metadata", {}))
        metadata.update({"original_task_type": "counting", "counting_unresolved": True})
        updated["metadata"] = metadata
    if "instruction" in updated:
        updated["instruction"] = strengthen_instruction(task, str(updated["instruction"]))
    if "answer" in updated:
        updated["answer"] = strengthen_answer(task, updated["answer"])
    messages = updated.get("messages")
    if isinstance(messages, list):
        rewritten: list[dict[str, Any]] = []
        for message in messages:
            copy = dict(message)
            content = copy.get("content")
            if copy.get("role") == "user" and isinstance(content, list):
                copy["content"] = [
                    {
                        **dict(item),
                        "text": strengthen_instruction(task, str(item.get("text", ""))),
                    }
                    if item.get("type") == "text"
                    else dict(item)
                    for item in content
                ]
            elif copy.get("role") == "assistant":
                copy["content"] = strengthen_answer(task, content)
            rewritten.append(copy)
        updated["messages"] = rewritten
    return updated, unresolved


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        raise SystemExit(f"Input JSONL does not exist: {source}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {output}")
    rows: list[dict[str, Any]] = []
    unresolved_ids: list[str] = []
    for row in read_jsonl(source):
        rewritten, unresolved = rewrite_row(row)
        rows.append(rewritten)
        if unresolved:
            unresolved_ids.append(str(row.get("id", "<unknown>")))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    print(
        f"Wrote {len(rows)} rows to {output}; unresolved_counting={len(unresolved_ids)}; "
        f"sample_ids={unresolved_ids[:20]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
