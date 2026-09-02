"""Coverage audit for TaskGraph target categories in local datasets."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from sat_rs_vlm.semantics.mentions import term_mentions
from sat_rs_vlm.semantics.ontology import load_ontology

from .capabilities import TargetCapabilityClassifier

_TARGET_PATTERN = re.compile(r'T\(\s*["\']([^"\']+)["\']')


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            yield value


def _read_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        return _read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        return payload
    raise ValueError(f"{path} must contain a JSON object list or records list")


def _row_text(row: Mapping[str, Any]) -> str:
    values = [row.get(key) for key in ("question", "instruction", "Text", "text")]
    messages = row.get("messages")
    if isinstance(messages, list):
        values.extend(
            message.get("content")
            for message in messages
            if isinstance(message, Mapping) and message.get("role") in {"user", "assistant"}
        )
    return "\n".join(str(value) for value in values if isinstance(value, str))


def _target_strings(row: Mapping[str, Any], text: str) -> list[tuple[str, str]]:
    targets = [(match, "dsl_target") for match in _TARGET_PATTERN.findall(text)]
    targets.extend(
        (str(value), "target_field")
        for key in ("target_category", "target")
        for value in ([row.get(key)] if not isinstance(row.get(key), list) else row.get(key, []))
        if isinstance(value, str) and value.strip()
    )
    return targets


def build_target_capability_coverage(
    input_paths: Iterable[str | Path],
    *,
    ontology_path: str | Path,
) -> dict[str, Any]:
    """Build category/frequency/capability counts from local source rows."""

    ontology_file = Path(ontology_path).expanduser().resolve()
    ontology = load_ontology(ontology_file)
    classifier = TargetCapabilityClassifier(ontology)
    counters: Counter[tuple[str, str | None, str]] = Counter()
    source_kinds: dict[tuple[str, str | None, str], set[str]] = {}
    source_rows: Counter[str] = Counter()
    source_targets: Counter[str] = Counter()
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            continue
        row_count = 0
        for row in _read_rows(path):
            row_count += 1
            text = _row_text(row)
            targets = _target_strings(row, text)
            if not targets:
                mentions = term_mentions(text.casefold(), ontology["objects"])
                targets = [(mention.alias, "ontology_alias") for mention in mentions]
            for requested, source_kind in targets:
                decision = classifier.classify(requested)
                counter_key = (
                    requested,
                    decision.canonical_category,
                    decision.capability.value,
                )
                counters[counter_key] += 1
                source_kinds.setdefault(counter_key, set()).add(source_kind)
                source_targets[str(path)] += 1
        source_rows[str(path)] += row_count
    categories: list[dict[str, Any]] = []
    for (requested, canonical, capability), frequency in sorted(
        counters.items(), key=lambda item: (-item[1], item[0])
    ):
        categories.append(
            {
                "target_category": requested,
                "canonical_category": canonical,
                "capability": capability,
                "frequency": frequency,
                "source_kinds": sorted(source_kinds[(requested, canonical, capability)]),
            }
        )
    capability_counts = Counter(
        item["capability"] for item in categories for _ in range(int(item["frequency"]))
    )
    return {
        "schema_version": "taskgraph-target-capability-coverage-v1",
        "ontology_path": str(ontology_file),
        "ontology_version": ontology.get("ontology_version"),
        "unresolved_policy": classifier.unresolved_policy,
        "source_rows": dict(sorted(source_rows.items())),
        "source_target_mentions": dict(sorted(source_targets.items())),
        "target_category_count": len(categories),
        "target_mention_count": sum(item["frequency"] for item in categories),
        "capability_frequency": dict(sorted(capability_counts.items())),
        "categories": categories,
    }


def write_target_capability_coverage(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = ["build_target_capability_coverage", "write_target_capability_coverage"]
