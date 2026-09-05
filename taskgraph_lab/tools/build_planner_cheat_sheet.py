"""Build a leakage-safe hard-example bank from validated Planner train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from taskgraph_lab.retrieval import HARD_INTENTS
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.validator import validate_candidate
from taskgraph_lab.training.planner_dataset import file_sha256

DEFAULT_QUOTAS = {
    "RELATIONAL_COUNT": 100,
    "OBJECT_RELATION": 80,
    "ROUTE_PLANNING": 40,
    "COMPLEX_REASONING": 20,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _messages(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    messages = row.get("messages") or []
    users = [message for message in messages if message.get("role") == "user"]
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError(f"{row.get('id')}: expected one user and one assistant message")
    payload = json.loads(str(users[0].get("content", "")))
    if not isinstance(payload, dict):
        raise TypeError(f"{row.get('id')}: user payload must be an object")
    return payload, str(assistants[0].get("content", "")).strip()


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.startswith("$n") else []
    if isinstance(value, list):
        return [reference for item in value for reference in _references(item)]
    if isinstance(value, dict):
        return [reference for item in value.values() for reference in _references(item)]
    return []


def _example_metadata(graph: dict[str, Any]) -> dict[str, Any]:
    depth: dict[str, int] = {}
    relation_depth: dict[str, int] = {}
    relations: list[str] = []
    operators: list[str] = []
    ordinal_signal = False
    rank_signal = False
    for node in graph["nodes"]:
        references = _references(node.get("inputs") or {})
        parents = [reference[1:] for reference in references]
        depth[node["id"]] = 1 + max((depth.get(parent, 0) for parent in parents), default=0)
        relation_step = int(
            node["op"] == "RELATION"
            or (node["op"] == "SELECT" and node.get("params", {}).get("mode") == "RELATION")
        )
        relation_depth[node["id"]] = relation_step + max(
            (relation_depth.get(parent, 0) for parent in parents), default=0
        )
        operators.append(str(node["op"]))
        relation = node.get("params", {}).get("relation")
        if relation is not None:
            relations.append(str(relation))
        mode = node.get("params", {}).get("mode")
        ordinal_signal = ordinal_signal or mode == "ORDINAL"
        rank_signal = rank_signal or mode in {"RANK", "EXTREME"}
    return {
        "relation_depth": max(relation_depth.values(), default=0),
        "graph_depth": max(depth.values(), default=0),
        "node_count": len(graph["nodes"]),
        "relations": sorted(set(relations)),
        "operators": operators,
        "has_bbox": "REGION_FROM_BBOX" in operators,
        "has_marker": "FIND_MARKER" in operators,
        "multi_image": sum(
            1
            for reference in {
                item
                for node in graph["nodes"]
                for item in (node.get("inputs") or {}).values()
                if isinstance(item, str) and item.startswith("$image")
            }
        )
        > 1,
        "ordinal_signal": ordinal_signal,
        "rank_signal": rank_signal,
    }


def _bucket(node_count: int) -> str:
    if node_count >= 10:
        return "10_plus"
    if node_count >= 7:
        return "7_9"
    if node_count >= 4:
        return "4_6"
    return "short"


def _select(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in candidates:
        groups[_bucket(int(example["metadata"]["node_count"]))].append(example)
    for values in groups.values():
        values.sort(
            key=lambda example: hashlib.sha256(
                str(example["example_id"]).encode("utf-8")
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    order = ("10_plus", "7_9", "4_6", "short")
    while len(selected) < limit and any(groups.values()):
        for name in order:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
    return selected


def build_bank(
    *,
    train_file: Path,
    heldout_files: list[Path],
    output_dir: Path,
    rule_cards: Path,
    quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_file)
    heldout_ids = {
        str(row.get("id") or row.get("sample_id"))
        for path in heldout_files
        for row in _read_jsonl(path)
    }
    train_ids = [str(row.get("id")) for row in train_rows]
    overlap = sorted(set(train_ids) & heldout_ids)
    if overlap:
        raise ValueError(f"cheat-sheet train/heldout overlap: {overlap[:50]}")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("cheat-sheet train file contains duplicate sample ids")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        metadata = row.get("metadata") or {}
        if metadata.get("source_bucket") != "accepted":
            continue
        intent = str(metadata.get("intent") or "")
        if intent not in HARD_INTENTS:
            continue
        payload, dsl = _messages(row)
        lowered = parse_taskgraph_dsl_payload(dsl)
        target, validation = validate_candidate(
            lowered,
            inputs=payload.get("inputs") or {},
            question=str(payload.get("question", "")),
            question_type=str(payload.get("question_type", "FREE_FORM")),
        )
        if target is None or not validation.valid:
            continue
        grammar = CanonicalDSLPrefixGrammar((payload.get("inputs") or {}).keys())
        if not grammar.accepts(dsl):
            continue
        canonical = canonicalize_target(target)
        if compile_taskgraph_to_dsl(canonical) != dsl:
            continue
        sample_id = str(row["id"])
        candidates[intent].append(
            {
                "example_id": sample_id,
                "source_split": "train",
                "intent": intent,
                "question": str(payload["question"]),
                "metadata": _example_metadata(canonical),
                "dsl": dsl,
            }
        )

    selected = [
        example
        for intent, limit in (quotas or DEFAULT_QUOTAS).items()
        for example in _select(candidates[intent], limit)
    ]
    selected.sort(key=lambda example: (str(example["intent"]), str(example["example_id"])))
    if set(example["example_id"] for example in selected) & heldout_ids:
        raise AssertionError("heldout leakage detected after selection")
    output_dir.mkdir(parents=True, exist_ok=False)
    bank_path = output_dir / "examples.jsonl"
    with bank_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in selected:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    copied_rules = output_dir / "rule_cards.txt"
    copied_rules.write_text(rule_cards.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "version": "taskgraph-hard-cheat-sheet-v1",
        "train_file": str(train_file.resolve()),
        "train_sha256": file_sha256(train_file),
        "heldout_files": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in heldout_files
        ],
        "train_heldout_overlap_count": 0,
        "candidate_count": sum(map(len, candidates.values())),
        "selected_count": len(selected),
        "candidate_intent_distribution": dict(
            sorted((intent, len(values)) for intent, values in candidates.items())
        ),
        "selected_intent_distribution": dict(
            sorted(Counter(example["intent"] for example in selected).items())
        ),
        "selected_node_bucket_distribution": dict(
            sorted(
                Counter(
                    _bucket(example["metadata"]["node_count"])
                    for example in selected
                ).items()
            )
        ),
        "example_ids": [example["example_id"] for example in selected],
        "bank_path": str(bank_path.resolve()),
        "bank_sha256": file_sha256(bank_path),
        "rule_cards_path": str(copied_rules.resolve()),
        "rule_cards_sha256": file_sha256(copied_rules),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--heldout-file", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rule-cards",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rag/hard_rule_cards.txt",
    )
    args = parser.parse_args()
    manifest = build_bank(
        train_file=args.train_file,
        heldout_files=args.heldout_file,
        output_dir=args.output_dir,
        rule_cards=args.rule_cards,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
