"""Build explainable MME LOCATE regression and replay artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _contains_locate(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_locate(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_locate(item) for item in value)
    return isinstance(value, str) and bool(re.search(r"\bLOCATE\b", value))


def _is_wrong_locate(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "success"
        and isinstance(row.get("answer_judgment"), Mapping)
        and row["answer_judgment"].get("status") == "incorrect"
        and _contains_locate(row.get("reasoning_chain", {}))
    )


def _taskgraph(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    chain = row.get("reasoning_chain")
    if isinstance(chain, Mapping):
        planner = chain.get("planner")
        if isinstance(planner, Mapping) and isinstance(planner.get("taskgraph"), Mapping):
            return planner["taskgraph"]
    trace = row.get("trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("taskgraph"), Mapping):
        return trace["taskgraph"]
    return None


def _locate_traces(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = row.get("trace")
    if not isinstance(trace, Mapping):
        return []
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [dict(node) for node in nodes if isinstance(node, Mapping) and node.get("operator") == "LOCATE"]


def _option_body(option: Any) -> str:
    return re.sub(r"^\s*[\(\[\{]?\s*[A-Za-z]+\s*[\)\]\}:.\-]?\s*", "", str(option)).strip()


def _position_pattern(row: Mapping[str, Any]) -> dict[str, Any]:
    graph = _taskgraph(row) or {}
    options = graph.get("choices", [])
    if not isinstance(options, list):
        options = []
    bodies = [_option_body(option) for option in options]
    lowered = [body.casefold() for body in bodies]
    return {
        "task_category": row.get("task_category"),
        "question_type": graph.get("question_type"),
        "option_count": len(options),
        "has_absence_option": any(
            "doesn't feature" in body or "does not feature" in body or "no position" in body
            for body in lowered
        ),
        "position_option_ids": [
            chr(ord("A") + index)
            for index, body in enumerate(lowered)
            if re.search(
                r"left|right|upper|lower|top|bottom|middle|center|centre|corner|edge|north|south|east|west",
                body,
            )
        ],
        "option_bodies": bodies,
    }


def _replay_sample(row: Mapping[str, Any]) -> dict[str, Any] | None:
    graph = _taskgraph(row)
    if graph is None:
        return None
    question = graph.get("question")
    choices = graph.get("choices")
    image_paths = row.get("input_image_paths", [])
    if not isinstance(question, str) or not isinstance(choices, list) or not isinstance(image_paths, list):
        return None
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset", "MME_RealWorld_RS"),
        "task_category": row.get("task_category", "position"),
        "question": question,
        "choices": choices,
        "question_type": graph.get("question_type", "MULTIPLE_CHOICE_SINGLE"),
        "image_paths": image_paths,
        "reference_answer": row.get("answer"),
        "graph": graph,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolution_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    fallbacks = Counter[str]()
    for row in rows:
        for locate in _locate_traces(row):
            metadata = locate.get("trace_metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            status = metadata.get("final_resolution_status")
            if status is not None:
                statuses[str(status)] += 1
            reason = metadata.get("fallback_reason")
            if reason:
                fallbacks[str(reason)] += 1
    return {"final_resolution_status": dict(sorted(statuses.items())), "fallback_reason": dict(sorted(fallbacks.items()))}


def _regression_report(before: list[dict[str, Any]], after: list[dict[str, Any]] | None) -> dict[str, Any]:
    selected = [row for row in before if _is_wrong_locate(row)]
    report: dict[str, Any] = {
        "status": "pending_after_run" if after is None else "complete",
        "selection_rule": {
            "status": "success",
            "answer_judgment.status": "incorrect",
            "reasoning_chain_contains": "LOCATE",
        },
        "baseline_selected_count": len(selected),
        "baseline_resolution_stats": _resolution_stats(selected),
    }
    if after is None:
        return report
    baseline_by_id = {str(row.get("sample_id")): row for row in selected}
    after_by_id = {str(row.get("sample_id")): row for row in after}
    comparisons: list[dict[str, Any]] = []
    for sample_id, baseline in baseline_by_id.items():
        candidate = after_by_id.get(sample_id)
        if candidate is None:
            comparisons.append({"sample_id": sample_id, "status": "missing_after"})
            continue
        before_answer = baseline.get("answer_judgment", {}).get("status")
        after_judgment = candidate.get("answer_judgment", {})
        comparisons.append(
            {
                "sample_id": sample_id,
                "before_status": before_answer,
                "after_status": after_judgment.get("status"),
                "before_answer": baseline.get("answer"),
                "after_answer": candidate.get("answer"),
                "after_result_status": candidate.get("status"),
                "after_elapsed_ms": candidate.get("elapsed_ms"),
            }
        )
    report.update(
        {
            "after_selected_count": sum(_is_wrong_locate(row) for row in after),
            "after_resolution_stats": _resolution_stats(after),
            "comparison": comparisons,
            "after_status_counts": dict(
                sorted(Counter(item.get("after_status", "missing_after") for item in comparisons).items())
            ),
        }
    )
    return report


def build_reports(input_path: Path, output_dir: Path, after_path: Path | None = None) -> None:
    rows = _read_jsonl(input_path)
    selected = [row for row in rows if _is_wrong_locate(row)]
    cases = []
    replay_rows = []
    for row in selected:
        case = {
            "sample_id": row.get("sample_id"),
            "task_category": row.get("task_category"),
            "question": (_taskgraph(row) or {}).get("question"),
            "answer": row.get("answer"),
            "reference_answer": row.get("answer_judgment", {}).get("reference_answer"),
            "answer_judgment": row.get("answer_judgment"),
            "input_image_paths": row.get("input_image_paths", []),
            "intermediate_output_paths": row.get("intermediate_output_paths", []),
            "locate_traces": _locate_traces(row),
            "taskgraph": _taskgraph(row),
            "row": row,
        }
        cases.append(case)
        replay = _replay_sample(row)
        if replay is not None:
            replay_rows.append(replay)

    patterns = Counter()
    pattern_examples: dict[str, list[str]] = {}
    for row in rows:
        pattern = _position_pattern(row)
        key = json.dumps(
            {
                "task_category": pattern["task_category"],
                "option_count": pattern["option_count"],
                "has_absence_option": pattern["has_absence_option"],
                "position_option_ids": pattern["position_option_ids"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        patterns[key] += 1
        pattern_examples.setdefault(key, []).append(str(row.get("sample_id")))

    _write_json(
        output_dir / "mme_locate_wrong_cases.json",
        {
            "source": str(input_path),
            "selection_rule": {
                "status": "success",
                "answer_judgment.status": "incorrect",
                "reasoning_chain_contains": "LOCATE",
            },
            "selected_count": len(cases),
            "resolution_stats": _resolution_stats(selected),
            "cases": cases,
        },
    )
    _write_json(
        output_dir / "mme_position_choice_patterns.json",
        {
            "source": str(input_path),
            "row_count": len(rows),
            "patterns": [
                {
                    **json.loads(key),
                    "count": count,
                    "sample_ids": pattern_examples[key][:20],
                }
                for key, count in sorted(patterns.items())
            ],
        },
    )
    after_rows = _read_jsonl(after_path) if after_path is not None else None
    _write_json(output_dir / "mme_locate_regression_after.json", _regression_report(rows, after_rows))
    replay_path = output_dir / "mme_locate_regression_replay.jsonl"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in replay_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_count": len(cases),
                "replay_count": len(replay_rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--after-input", type=Path)
    args = parser.parse_args()
    build_reports(args.input, args.output_dir, args.after_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
