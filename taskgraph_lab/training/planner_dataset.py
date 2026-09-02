"""Strict loader and deterministic split writer for Planner SFT records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from taskgraph_lab import PROMPT_VERSION
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target, stable_json_dumps
from taskgraph_lab.taskgraph.dsl import (
    DSL_VERSION,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
from taskgraph_lab.taskgraph.validator import validate_candidate

TargetFormat = Literal["dsl", "json"]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            yield payload


def _choice_question_type(sample: Mapping[str, Any]) -> str:
    choices = sample.get("choices")
    if isinstance(choices, list) and choices:
        # Cardinality belongs to final.answer_type and the question semantics.
        # The student never receives legacy SINGLE/MULTI transport labels.
        return "MULTIPLE_CHOICE"
    return str(sample.get("question_type", "FREE_FORM"))


class PlannerSFTDataset(Sequence[dict[str, Any]]):
    """Load accepted revalidation records without exposing source answers.

    Every record is revalidated against the current schema and DSL round trip.
    The emitted row is directly consumable by the repository's
    :class:`Qwen3VLDataset` after materialization.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        system_prompt: str | Path,
        target_format: TargetFormat = "dsl",
    ) -> None:
        self.source = Path(source).resolve()
        self.system_prompt_path = Path(system_prompt).resolve()
        if target_format not in {"dsl", "json"}:
            raise ValueError(f"unsupported Planner target format: {target_format}")
        self.target_format = target_format
        if not self.source.is_file():
            raise FileNotFoundError(f"accepted Planner source does not exist: {self.source}")
        if not self.system_prompt_path.is_file():
            raise FileNotFoundError(
                f"Planner system prompt does not exist: {self.system_prompt_path}"
            )
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()
        if not self.system_prompt:
            raise ValueError("Planner system prompt must not be empty")
        self._rows = [self._normalize_record(record) for record in _jsonl(self.source)]
        ids = [str(row["id"]) for row in self._rows]
        if len(ids) != len(set(ids)):
            duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
            raise ValueError(f"duplicate Planner sample ids: {duplicates}")
        if len(self._rows) < 2:
            raise ValueError("Planner SFT dataset requires at least two accepted records")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    def _normalize_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("accepted record is missing sample_id")
        if record.get("bucket") != "accepted":
            raise ValueError(f"{sample_id}: only bucket=accepted may enter Planner SFT")
        runtime_validation = record.get("runtime_validation")
        if (
            not isinstance(runtime_validation, Mapping)
            or runtime_validation.get("valid") is not True
        ):
            raise ValueError(f"{sample_id}: runtime validation is not valid")
        answer_audit = record.get("answer_audit")
        if not isinstance(answer_audit, Mapping) or answer_audit.get("valid") is not True:
            raise ValueError(f"{sample_id}: answer audit is not valid")
        sample = record.get("sample")
        taskgraph = record.get("taskgraph")
        if not isinstance(sample, Mapping) or not isinstance(taskgraph, Mapping):
            raise TypeError(f"{sample_id}: sample and taskgraph must be mappings")
        choices = sample.get("choices")
        if choices is not None and not isinstance(choices, list):
            raise TypeError(f"{sample_id}: choices must be a list or null")
        inputs = sample.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError(f"{sample_id}: inputs must be a non-empty mapping")
        question = str(sample.get("question", "")).strip()
        if not question:
            raise ValueError(f"{sample_id}: question must not be empty")
        question_type = _choice_question_type(sample)
        canonical = canonicalize_target(taskgraph)
        _, validation = validate_candidate(
            canonical,
            inputs=inputs,
            question=question,
            question_type=question_type,
        )
        if not validation.valid:
            codes = [issue.code for issue in validation.errors]
            raise ValueError(f"{sample_id}: current validation failed: {codes}")
        planner_dsl = compile_taskgraph_to_dsl(canonical)
        if canonicalize_target(parse_taskgraph_dsl(planner_dsl)) != canonical:
            raise ValueError(f"{sample_id}: Planner DSL round trip mismatch")
        user_payload = {
            "question": question,
            "question_type": question_type,
            "choices": choices,
            "inputs": dict(inputs),
        }
        assistant = planner_dsl if self.target_format == "dsl" else stable_json_dumps(canonical)
        metadata = dict(sample.get("metadata") or {})
        metadata.update(
            {
                "dataset": metadata.get("dataset"),
                "intent": canonical.get("intent"),
                "planner_target_format": self.target_format,
                "planner_dsl_version": DSL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "source_bucket": "accepted",
            }
        )
        return {
            "id": sample_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": stable_json_dumps(user_payload)},
                {"role": "assistant", "content": assistant},
            ],
            "task_type": "planner",
            "metadata": metadata,
        }

    def write_splits(
        self,
        output_dir: str | Path,
        *,
        validation_fraction: float = 0.1,
        seed: int = 42,
    ) -> dict[str, Any]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self._rows:
            intent = str(row["metadata"].get("intent") or "unknown")
            grouped.setdefault(intent, []).append(row)
        validation_ids: set[str] = set()
        for intent, rows in sorted(grouped.items()):
            ranked = sorted(
                rows,
                key=lambda row: (
                    hashlib.sha256(f"{seed}:{intent}:{row['id']}".encode()).hexdigest(),
                    str(row["id"]),
                ),
            )
            validation_count = (
                max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
                if len(ranked) > 1
                else 0
            )
            validation_ids.update(str(row["id"]) for row in ranked[:validation_count])
        splits = {
            "train": [row for row in self._rows if str(row["id"]) not in validation_ids],
            "validation": [row for row in self._rows if str(row["id"]) in validation_ids],
        }
        paths: dict[str, Path] = {}
        for split, rows in splits.items():
            path = output / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            paths[split] = path

        def split_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
            return dict(
                sorted(Counter(str(row["metadata"].get(key) or "unknown") for row in rows).items())
            )

        manifest = {
            "version": "taskgraph-planner-sft-v1",
            "source": str(self.source),
            "source_sha256": file_sha256(self.source),
            "system_prompt": str(self.system_prompt_path),
            "system_prompt_sha256": file_sha256(self.system_prompt_path),
            "prompt_version": PROMPT_VERSION,
            "target_format": self.target_format,
            "planner_dsl_version": DSL_VERSION,
            "split_seed": seed,
            "split_strategy": "stratified_by_intent_sha256",
            "validation_fraction": validation_fraction,
            "population_count": len(self),
            "splits": {
                split: {
                    "path": str(paths[split]),
                    "sha256": file_sha256(paths[split]),
                    "count": len(rows),
                    "sample_ids": [str(row["id"]) for row in rows],
                    "per_dataset": split_counts(rows, "dataset"),
                    "per_intent": split_counts(rows, "intent"),
                }
                for split, rows in splits.items()
            },
        }
        manifest_path = output / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = file_sha256(manifest_path)
        return manifest
