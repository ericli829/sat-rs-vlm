"""Build an independent Planner mixture without changing any source dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from taskgraph_lab import PROMPT_VERSION
from taskgraph_lab.taskgraph.dsl import (
    DSL_VERSION,
    DSLError,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.validator import validate_candidate
from taskgraph_lab.training.planner_dataset import file_sha256

SYNTHETIC_SOURCE = "synthetic_programmatic_plan"


def _read_rows(path: Path, *, synthetic: bool = False) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            raise ValueError(f"{path}:{line_number}: a nonempty id is required")
        messages = row.get("messages", [])
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{path}:{line_number}: expected system/user/assistant messages")
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError(f"{path}:{line_number}: Planner message contents must be strings")
        if row.get("task_type") != "planner":
            raise ValueError(f"{path}:{line_number}: task_type must be planner")
        if synthetic and row.get("metadata", {}).get("source") != SYNTHETIC_SOURCE:
            raise ValueError(f"{path}:{line_number}: synthetic source marker is missing")
        if synthetic and ("answer_audit" in row or "answer_audit" in row.get("metadata", {})):
            raise ValueError(f"{path}:{line_number}: synthetic plans cannot claim answer audit")
        # Historical teacher rows are preserved as supplied. Current stricter
        # validation failures are reported separately, never repaired silently.
        if synthetic:
            graph = parse_taskgraph_dsl(messages[2]["content"])
            user = json.loads(messages[1]["content"])
            if graph.final.answer_type.value != "BBOX":
                raise ValueError(f"{path}:{line_number}: synthetic final must be BBOX")
            if user.get("question_type") != "FREE_FORM" or user.get("choices") is not None:
                raise ValueError(f"{path}:{line_number}: synthetic bbox input must be FREE_FORM")
            _, validation = validate_candidate(
                graph,
                inputs=user["inputs"],
                question=user["question"],
                question_type=user["question_type"],
            )
            if not validation.valid:
                raise ValueError(f"{path}:{line_number}: synthetic graph validation failed")
        rows.append(row)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample ids")
    if not rows:
        raise ValueError(f"{path}: dataset must not be empty")
    if synthetic:
        questions = [_question_key(row) for row in rows]
        if len(questions) != len(set(questions)):
            raise ValueError(f"{path}: duplicate synthetic questions are not independent samples")
    return rows


def _question_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    original = metadata.get("original_question")
    if original is None:
        original = json.loads(row["messages"][1]["content"])["question"]
    original = str(original).split("\nResolve the referred target using localization", 1)[0]
    return " ".join(original.casefold().split())


def _group(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row["metadata"]
    return str(metadata.get("family", "unknown")), str(metadata.get("category", "unknown"))


def _rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _select_stratified(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Select without replacement using proportional family/category strata."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group(row)].append(row)
    quotas = {key: count * len(values) / len(rows) for key, values in groups.items()}
    allocated = {key: math.floor(quota) for key, quota in quotas.items()}
    remaining = count - sum(allocated.values())
    order = sorted(groups, key=lambda key: (-(quotas[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    selected = []
    for key, values in sorted(groups.items()):
        ranked = sorted(values, key=lambda row: (_rank(seed, str(row["id"])), str(row["id"])))
        selected.extend(ranked[: allocated[key]])
    return selected


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families: Counter[str] = Counter()
    finals: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    invalid_codes: Counter[str] = Counter()
    invalid_ids = []
    for row in rows:
        families[str(row.get("metadata", {}).get("family", "existing"))] += 1
        dsl = row["messages"][2]["content"]
        payload = parse_taskgraph_dsl_payload(dsl)
        finals[str(payload["final"]["answer_type"])] += 1
        operators.update(node["op"] for node in payload["nodes"])
        try:
            parse_taskgraph_dsl(dsl)
        except DSLError:
            invalid_ids.append(row["id"])
            user = json.loads(row["messages"][1]["content"])
            _, validation = validate_candidate(payload, inputs=user["inputs"])
            invalid_codes.update({issue.code for issue in validation.errors} or {"dsl_parse"})
    return {
        "families": dict(sorted(families.items())),
        "final_types": dict(sorted(finals.items())),
        "final_type_fractions": {name: count / len(rows) for name, count in sorted(finals.items())},
        "operator_nodes": dict(sorted(operators.items())),
        "current_dsl_validation": {
            "valid_count": len(rows) - len(invalid_ids),
            "invalid_count": len(invalid_ids),
            "failure_codes_rows": dict(sorted(invalid_codes.items())),
            "invalid_sample_ids": invalid_ids,
            "historical_rows_preserved_without_repair": True,
        },
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_mixture(
    *,
    base_train: Path,
    base_test: Path,
    synthetic_train: Path,
    synthetic_validation: Path,
    output_dir: Path,
    synthetic_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0.0 < synthetic_fraction < 1.0:
        raise ValueError("synthetic_fraction must be strictly between zero and one")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"choose a new output directory: {output_dir}")
    original = _read_rows(base_train)
    heldout = _read_rows(base_test)
    extra = _read_rows(synthetic_train, synthetic=True)
    validation = _read_rows(synthetic_validation, synthetic=True)
    original_ids = {row["id"] for row in original}
    heldout_ids = {row["id"] for row in heldout}
    extra_ids = {row["id"] for row in extra}
    validation_ids = {row["id"] for row in validation}
    if original_ids & heldout_ids:
        raise ValueError("base train and fixed test ids overlap")
    if extra_ids & (original_ids | heldout_ids | validation_ids):
        raise ValueError("synthetic train ids overlap another source")
    if validation_ids & (original_ids | heldout_ids):
        raise ValueError("synthetic validation ids overlap an existing source")
    train_templates = {
        row["metadata"]["template_family"]
        for row in extra
        if row["metadata"].get("template_family")
    }
    validation_templates = {
        row["metadata"]["template_family"]
        for row in validation
        if row["metadata"].get("template_family")
    }
    if train_templates & validation_templates:
        raise ValueError("synthetic train and validation template families overlap")
    protected_questions = {_question_key(row) for row in heldout + validation}
    if any(_question_key(row) in protected_questions for row in extra):
        raise ValueError("synthetic training questions overlap a protected validation/test set")
    if {_question_key(row) for row in validation} & {_question_key(row) for row in original}:
        raise ValueError("synthetic validation questions overlap existing training questions")
    requested_count = round(len(original) * synthetic_fraction / (1.0 - synthetic_fraction))
    if requested_count < 1:
        raise ValueError("requested fraction rounds to zero synthetic rows")
    if requested_count > len(extra):
        raise ValueError(
            f"requested {requested_count} distinct synthetic rows but only {len(extra)} available; "
            "reduce synthetic_fraction or generate more data; repetition is forbidden"
        )
    selected = _select_stratified(extra, requested_count, seed)
    combined = sorted(original + selected, key=lambda row: (_rank(seed, str(row["id"])), row["id"]))
    distributions = {
        "base_train": _distribution(original),
        "synthetic_selected": _distribution(selected),
        "train": _distribution(combined),
    }
    # Every check above runs before creating the new directory or writing any file.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_rows(output_dir / "train.jsonl", combined)
    _write_rows(output_dir / "validation.jsonl", validation)
    shutil.copyfile(base_test, output_dir / "test.jsonl")
    test_hash = file_sha256(base_test)
    if file_sha256(output_dir / "test.jsonl") != test_hash:
        raise RuntimeError("fixed held-out test byte preservation failed")
    sources = {}
    for name, path, rows in (
        ("base_train", base_train, original),
        ("base_test", base_test, heldout),
        ("synthetic_train", synthetic_train, extra),
        ("synthetic_validation", synthetic_validation, validation),
    ):
        sources[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "count": len(rows),
        }
    manifest = {
        "version": "planner-synthetic-mixture-v1",
        "target_format": "dsl",
        "planner_dsl_version": DSL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "seed": seed,
        "population_count": len(combined) + len(validation) + len(heldout),
        "training_population_count": len(combined),
        "base_training_count": len(original),
        "synthetic_available_count": len(extra),
        "synthetic_selected_count": len(selected),
        "requested_synthetic_fraction": synthetic_fraction,
        "actual_synthetic_fraction": len(selected) / len(combined),
        "sampling": "without_replacement_stratified_family_category_sha256",
        "sources": sources,
        "fixed_test_sha256": test_hash,
        "fixed_test_preserved_bytes": True,
        "existing_train_test_exact_question_overlap": len(
            {_question_key(row) for row in original} & {_question_key(row) for row in heldout}
        ),
        "selected_synthetic_ids": [row["id"] for row in selected],
        "distributions": distributions,
        "base_current_dsl_validation": distributions["base_train"]["current_dsl_validation"],
        "splits": {},
    }
    for name, rows in (("train", combined), ("validation", validation), ("test", heldout)):
        path = output_dir / f"{name}.jsonl"
        manifest["splits"][name] = {
            "path": str(path),
            "count": len(rows),
            "sha256": file_sha256(path),
        }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-train",
        "base-test",
        "synthetic-train",
        "synthetic-validation",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--synthetic-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_mixture(**vars(args))
    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "manifest": str(args.output_dir.resolve() / "dataset_manifest.json"),
        "base_training_count": manifest["base_training_count"],
        "synthetic_selected_count": manifest["synthetic_selected_count"],
        "training_population_count": manifest["training_population_count"],
        "requested_synthetic_fraction": manifest["requested_synthetic_fraction"],
        "actual_synthetic_fraction": manifest["actual_synthetic_fraction"],
        "validation_count": manifest["splits"]["validation"]["count"],
        "fixed_test_count": manifest["splits"]["test"]["count"],
        "fixed_test_preserved_bytes": manifest["fixed_test_preserved_bytes"],
        "base_current_dsl_invalid_count": manifest["base_current_dsl_validation"]["invalid_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
