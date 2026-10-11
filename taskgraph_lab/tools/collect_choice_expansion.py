"""Audit Codex choice plans and build an independent, provenance-preserving SFT release.

Only JSONL plans are read; this module never executes plan-generator scripts or
calls a teacher API. Ground-truth labels are used only by the existing answer
cardinality audit and never enter Planner messages.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from taskgraph_lab import PROMPT_VERSION, SCHEMA_VERSION
from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.quality.answer_audit import audit_choice_answer, load_answer_index
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    DSL_VERSION,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.dsl.constraint import CanonicalDSLPrefixGrammar
from taskgraph_lab.taskgraph.validator import validate_candidate
from taskgraph_lab.tools.choice_source_inputs import normalize_hrscd_inputs
from taskgraph_lab.training.planner_dataset import PlannerSFTDataset, file_sha256

LAB_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "codex_subagent_choice_plan"
REVIEW_SCOPE = "text-only logical plan; no visual answer verification"
ANSWER_KEYS = {"answer", "source_answer", "ground truth", "ground_truth", "selected_choice_labels"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: JSON object required")
        records.append(row)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _contains_answer_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in ANSWER_KEYS or _contains_answer_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_answer_key(item) for item in value)
    return False


def _question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def _row_user(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError(f"{row.get('id')}: expected system/user/assistant messages")
    if any(not isinstance(item.get("content"), str) for item in messages):
        raise ValueError(f"{row.get('id')}: Planner messages must be text")
    user = json.loads(messages[1]["content"])
    if not isinstance(user, dict) or not user.get("choices"):
        raise ValueError(f"{row.get('id')}: expected a choice-question input")
    if _contains_answer_key(user):
        raise ValueError(f"{row.get('id')}: answer field leaked into Planner input")
    return user


def _normalized_seed(row: dict[str, Any]) -> NormalizedSample:
    if _contains_answer_key(row):
        raise ValueError(f"{row.get('sample_id')}: prepared seed must not contain ground truth")
    payload = dict(row)
    payload.setdefault("question_type", "MULTIPLE_CHOICE")
    sample = NormalizedSample.model_validate(payload)
    if not sample.choices:
        raise ValueError(f"{sample.sample_id}: choice options are required")
    return sample


def _production_validation(
    dsl: str,
    *,
    question: str,
    choices: list[str] | None,
    inputs: Mapping[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    """Exercise the actual text-to-production bridge without loading a model."""
    from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
    from sat_rs_vlm.taskgraph.runtime_types import ImageRef

    try:
        request = PlannerRequest(
            question=question,
            question_type="MULTIPLE_CHOICE",
            choices=tuple(choices or []),
            inputs={
                "$" + str(key).removeprefix("$"): ImageRef(
                    str(spec["uri_or_key"] if isinstance(spec, Mapping) else spec.uri_or_key)
                )
                for key, spec in inputs.items()
            },
            sample_id=sample_id,
        )
        graph = Qwen3VLPlannerProvider._to_production_graph(dsl, request)
        return {
            "valid": True,
            "bridge": "Qwen3VLPlannerProvider._to_production_graph",
            "node_count": len(graph.nodes),
            "input_names": sorted(graph.inputs),
            "model_loaded": False,
        }
    except Exception as exc:
        return {
            "valid": False,
            "bridge": "Qwen3VLPlannerProvider._to_production_graph",
            "errors": ["production_graph_rejected"],
            "detail": f"{type(exc).__name__}: {exc}",
            "model_loaded": False,
        }


def _audit_dsl(dsl: str, sample: NormalizedSample) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        graph = parse_taskgraph_dsl(dsl)
        canonical = canonicalize_target(graph)
        _, validation = validate_candidate(
            canonical,
            inputs=sample.inputs,
            question=sample.question,
            question_type="MULTIPLE_CHOICE",
        )
        if not validation.valid:
            return None, {"valid": False, "errors": [issue.code for issue in validation.errors]}
        if canonical["final"]["answer_type"] not in {"CHOICE_SINGLE", "CHOICE_MULTI"}:
            return None, {"valid": False, "errors": ["choice_final_required"]}
        compiled = compile_taskgraph_to_dsl(canonical)
        if canonicalize_target(parse_taskgraph_dsl(compiled)) != canonical:
            return None, {"valid": False, "errors": ["dsl_roundtrip_mismatch"]}
        production = _production_validation(
            compiled,
            question=sample.question,
            choices=sample.choices,
            inputs=sample.inputs,
            sample_id=sample.sample_id,
        )
        if not production["valid"]:
            return None, {
                "valid": False,
                "errors": ["production_graph_rejected"],
                "production_validation": production,
            }
        prefix = CanonicalDSLPrefixGrammar(image_refs=[f"${key}" for key in sample.inputs]).analyze(
            compiled
        )
        if not prefix.valid_prefix or not prefix.complete:
            return None, {
                "valid": False,
                "errors": ["constraint_grammar_rejected"],
                "reason": prefix.reason,
            }
        semantics = []
        for node in canonical["nodes"]:
            if node["op"] == "SELECT" and node["params"]["mode"] == "SEMANTIC":
                semantics.append(
                    {
                        "node": node["id"],
                        "query": node["params"]["query"],
                        "selection_type": node["params"]["selection_type"],
                        "scope": node["inputs"].get("scope"),
                        "reference": node["inputs"].get("reference"),
                    }
                )
        return canonical, {
            "valid": True,
            "runtime_validation": validation.model_dump(mode="json"),
            "canonical_dsl": compiled,
            "dsl_roundtrip_valid": True,
            "constraint_complete": True,
            "production_validation": production,
            "semantic_selects": semantics,
        }
    except Exception as exc:
        return None, {
            "valid": False,
            "errors": ["dsl_or_validation_error"],
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _historical_audit(row: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = parse_taskgraph_dsl_payload(row["messages"][2]["content"])
        _, validation = validate_candidate(
            payload,
            inputs=user["inputs"],
            question=user["question"],
            question_type="MULTIPLE_CHOICE",
        )
        parse_taskgraph_dsl(row["messages"][2]["content"])
        errors = [issue.code for issue in validation.errors]
        production = _production_validation(
            row["messages"][2]["content"],
            question=user["question"],
            choices=user["choices"],
            inputs=user["inputs"],
            sample_id=str(row["id"]),
        )
        if not production["valid"]:
            errors.append("production_graph_rejected")
        return {
            "valid": validation.valid and production["valid"],
            "errors": errors,
            "production_validation": production,
        }
    except Exception as exc:
        return {
            "valid": False,
            "errors": ["current_dsl_rejected"],
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _manifest_ids(prepared: Path) -> list[str] | None:
    for name in ("selection_manifest.json", "manifest.json", "dataset_manifest.json"):
        path = prepared / name
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        for key in ("selected_sample_ids", "selected_new_ids", "sample_ids"):
            if key in manifest:
                value = manifest[key]
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise ValueError(f"{path}: {key} must be a list of sample IDs")
                return value
    return None


def collect(
    *,
    prepared_dir: Path,
    output_dir: Path,
    base_train: Path,
    fixed_test: Path,
    answer_index: dict[str, dict[str, Any]],
    target_count: int = 4500,
    validate_only: bool = False,
    system_prompt: Path | None = None,
) -> dict[str, Any]:
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    base_train, fixed_test = Path(base_train), Path(fixed_test)
    system_prompt = system_prompt or LAB_ROOT / "prompts/planner_student_system_prompt.txt"
    if not validate_only and (output_dir / "train.jsonl").exists():
        raise ValueError("refusing to overwrite an existing release train.jsonl")
    base_rows, test_rows = _read_jsonl(base_train), _read_jsonl(fixed_test)
    base_ids = [str(row["id"]) for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base train contains duplicate IDs")
    base_by_id = {str(row["id"]): row for row in base_rows}
    users = {str(row["id"]): _row_user(row) for row in base_rows}
    input_normalizations: dict[str, dict[str, Any]] = {}
    for sample_id, user in users.items():
        normalized = normalize_hrscd_inputs(user["inputs"])
        if normalized != user["inputs"]:
            input_normalizations[sample_id] = {
                "original_inputs": user["inputs"],
                "normalized_inputs": normalized,
                "reason": "explicit HRSCD paired image keys represent two temporal images",
            }
            user["inputs"] = normalized
    test_users = {str(row["id"]): _row_user(row) for row in test_rows}
    seed_samples = [_normalized_seed(row) for row in _read_jsonl(prepared_dir / "seed.jsonl")]
    seed_ids = [sample.sample_id for sample in seed_samples]
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("prepared seed contains duplicate IDs")
    expected = _manifest_ids(prepared_dir)
    if expected is not None and (
        len(expected) != len(set(expected)) or set(expected) != set(seed_ids)
    ):
        raise ValueError("prepared manifest IDs do not exactly match seed.jsonl")
    errors: list[dict[str, Any]] = []
    if len(base_rows) + len(seed_samples) != target_count:
        errors.append(
            {
                "code": "target_count_mismatch",
                "actual": len(base_rows) + len(seed_samples),
                "expected": target_count,
            }
        )
    base_questions = {_question_key(user["question"]) for user in users.values()}
    test_questions = {_question_key(user["question"]) for user in test_users.values()}
    question_ids: dict[str, list[str]] = {}
    for sample in seed_samples:
        key = _question_key(sample.question)
        question_ids.setdefault(key, []).append(sample.sample_id)
        if sample.sample_id in users or sample.sample_id in test_users:
            errors.append({"code": "new_id_leakage", "sample_id": sample.sample_id})
        if key in base_questions or key in test_questions:
            errors.append({"code": "new_question_leakage", "sample_id": sample.sample_id})
    for ids in question_ids.values():
        if len(ids) > 1:
            errors.append({"code": "duplicate_new_question", "sample_ids": ids})
    samples = {sample.sample_id: sample for sample in seed_samples}
    for sample_id, user in users.items():
        sample_metadata = {"dataset": base_by_id[sample_id].get("metadata", {}).get("dataset")}
        if sample_id in input_normalizations:
            normalization = input_normalizations[sample_id]
            sample_metadata.update(
                {
                    "original_inputs": normalization["original_inputs"],
                    "input_normalization_reason": normalization["reason"],
                }
            )
        samples[sample_id] = NormalizedSample.model_validate(
            {
                "sample_id": sample_id,
                "question": user["question"],
                "question_type": "MULTIPLE_CHOICE",
                "choices": user["choices"],
                "inputs": user["inputs"],
                "metadata": sample_metadata,
            }
        )
    plans: dict[str, dict[str, Any]] = {}
    plan_paths = sorted((prepared_dir / "plans").glob("*.jsonl"))
    for path in plan_paths:
        for plan in _read_jsonl(path):
            sample_id = str(plan.get("sample_id", ""))
            if sample_id in plans:
                errors.append({"code": "duplicate_plan", "sample_id": sample_id, "path": str(path)})
            elif sample_id not in samples:
                errors.append(
                    {"code": "out_of_bounds_plan", "sample_id": sample_id, "path": str(path)}
                )
            else:
                plans[sample_id] = plan
    missing = sorted(set(seed_ids) - set(plans))
    if missing:
        errors.append({"code": "missing_new_plans", "count": len(missing), "sample_ids": missing})
    accepted, audits = [], []
    semantic_counts: Counter[str] = Counter()
    for sample_id, plan in sorted(plans.items()):
        sample = samples[sample_id]
        if not isinstance(plan.get("dsl"), str) or not str(plan.get("rule_id", "")).strip():
            audits.append(
                {
                    "sample_id": sample_id,
                    "valid": False,
                    "errors": ["plan_dsl_and_rule_id_required"],
                }
            )
            errors.append({"code": "invalid_plan", "sample_id": sample_id})
            continue
        graph, audit = _audit_dsl(plan["dsl"], sample)
        audit.update(
            {
                "sample_id": sample_id,
                "revision": sample_id in users,
                "rule_id": plan["rule_id"],
                "review_notes": plan.get("review_notes"),
                "review_scope": REVIEW_SCOPE,
            }
        )
        answer_source = answer_index.get(sample_id)
        if graph is not None and answer_source is not None:
            if answer_source["choices"] != sample.choices:
                audit.update({"valid": False, "errors": ["source_choice_options_mismatch"]})
            else:
                answer_audit = audit_choice_answer(
                    answer=answer_source["answer"],
                    choices=answer_source["choices"],
                    final_answer_type=graph["final"]["answer_type"],
                )
                answer_audit["source"] = answer_source["source"]
                audit["answer_audit"] = answer_audit
                if answer_audit["valid"] is not True:
                    audit.update({"valid": False, "errors": ["answer_cardinality_audit_failed"]})
        elif graph is not None:
            audit.update({"valid": False, "errors": ["answer_source_missing"]})
        audits.append(audit)
        if not audit["valid"]:
            errors.append({"code": "invalid_plan", "sample_id": sample_id})
            continue
        metadata = {
            **sample.metadata,
            "source": SOURCE,
            "rule_id": plan["rule_id"],
            "review_scope": REVIEW_SCOPE,
            "visual_answer_verified": False,
            "review_notes": plan.get("review_notes"),
        }
        if _contains_answer_key(metadata):
            errors.append({"code": "metadata_answer_leakage", "sample_id": sample_id})
            continue
        payload = sample.model_dump(mode="json")
        payload["metadata"] = metadata
        accepted.append(
            {
                "sample_id": sample_id,
                "bucket": "accepted",
                "sample": payload,
                "taskgraph": graph,
                "runtime_validation": audit["runtime_validation"],
                "production_validation": audit["production_validation"],
                "answer_audit": audit["answer_audit"],
                "provider_trace": {
                    "provider": "codex_subagent",
                    "network_used": False,
                    "review_scope": REVIEW_SCOPE,
                },
                "planner_dsl": audit["canonical_dsl"],
            }
        )
        for selection in audit["semantic_selects"]:
            semantic_counts["nodes"] += 1
            semantic_counts[selection["selection_type"]] += 1
            semantic_counts["explicit_scope" if selection["scope"] else "implicit_scope"] += 1
            if selection["reference"]:
                semantic_counts["explicit_reference"] += 1
    historical = {
        str(row["id"]): _historical_audit(row, users[str(row["id"])]) for row in base_rows
    }
    revisions = {record["sample_id"] for record in accepted} & set(users)
    old_invalid = sorted(sample_id for sample_id, audit in historical.items() if not audit["valid"])
    remaining_invalid = sorted(set(old_invalid) - revisions)
    missing_normalized_revisions = sorted(set(input_normalizations) - revisions)
    if missing_normalized_revisions:
        errors.append(
            {
                "code": "normalized_input_requires_revision",
                "count": len(missing_normalized_revisions),
                "sample_ids": missing_normalized_revisions,
            }
        )
    if remaining_invalid:
        errors.append(
            {
                "code": "historical_current_validation_failed",
                "count": len(remaining_invalid),
                "sample_ids": remaining_invalid,
            }
        )
    report = {
        "version": "planner-choice-expansion-v1",
        "source": SOURCE,
        "review_scope": REVIEW_SCOPE,
        "valid": not errors,
        "validate_only": validate_only,
        "target_count": target_count,
        "base_count": len(base_rows),
        "selected_new_count": len(seed_samples),
        "received_new_plan_count": len(set(plans) & set(seed_ids)),
        "accepted_new_count": sum(record["sample_id"] in set(seed_ids) for record in accepted),
        "revision_count": len(revisions),
        "missing_new_ids": missing,
        "blocking_errors": errors,
        "semantic_select": dict(semantic_counts),
        "input_normalizations": input_normalizations,
        "historical_current_validation": {
            "invalid_count": len(old_invalid),
            "invalid_ids": old_invalid,
            "invalid_records": {sample_id: historical[sample_id] for sample_id in old_invalid},
            "remaining_invalid_count": len(remaining_invalid),
            "remaining_invalid_ids": remaining_invalid,
            "all_release_rows_current_valid": not remaining_invalid and not errors,
        },
        "leakage": {
            "historical_question_overlap_count": len(base_questions & test_questions),
            "historical_id_overlap_count": len(set(users) & set(test_users)),
            "new_cross_split_allowed": False,
        },
        "source_sha256": {
            "seed": file_sha256(prepared_dir / "seed.jsonl"),
            "base_train": file_sha256(base_train),
            "fixed_test": file_sha256(fixed_test),
            "system_prompt": file_sha256(system_prompt),
            "plans": {path.name: file_sha256(path) for path in plan_paths},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "plan_audit.jsonl", audits)
    _write_jsonl(output_dir / "accepted_plans.jsonl", accepted)
    (output_dir / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if validate_only or errors:
        return report
    dataset = PlannerSFTDataset(
        output_dir / "accepted_plans.jsonl", system_prompt=system_prompt, target_format="dsl"
    )
    generated = {str(row["id"]): row for row in dataset}
    new_rows = [generated[sample_id] for sample_id in seed_ids]
    release_rows = [generated.get(str(row["id"]), row) for row in base_rows] + new_rows
    # Preserve every untouched base row byte-for-byte, including its old prompt.
    raw_base = [line for line in base_train.read_bytes().splitlines(keepends=True) if line.strip()]
    with (output_dir / "train.jsonl").open("wb") as handle:
        for row, raw in zip(base_rows, raw_base, strict=True):
            if str(row["id"]) in revisions:
                revised = generated[str(row["id"])]
                # Revisions may introduce SELECT_SEM, which the old system
                # prompt explicitly omitted from its allowed node surfaces.
                # Keep the current prompt supplied by PlannerSFTDataset.
                revised["messages"][1] = row["messages"][1]
                if str(row["id"]) in input_normalizations:
                    revised["messages"][1] = {
                        "role": "user",
                        "content": json.dumps(
                            users[str(row["id"])],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                handle.write(
                    (json.dumps(revised, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )
            else:
                handle.write(raw if raw.endswith(b"\n") else raw + b"\n")
        for row in new_rows:
            handle.write(
                (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            )
    _write_jsonl(output_dir / "new_train.jsonl", new_rows)
    shutil.copyfile(fixed_test, output_dir / "test.jsonl")
    final_types = Counter(
        parse_taskgraph_dsl_payload(row["messages"][2]["content"])["final"]["answer_type"]
        for row in release_rows
    )
    manifest = {
        **report,
        "population_count": len(release_rows),
        "target_format": "dsl",
        "planner_dsl_version": DSL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "final_types": dict(final_types),
        "fixed_test_preserved_bytes": file_sha256(fixed_test)
        == file_sha256(output_dir / "test.jsonl"),
        "splits": {
            "train": {
                "path": str((output_dir / "train.jsonl").resolve()),
                "count": len(release_rows),
                "sha256": file_sha256(output_dir / "train.jsonl"),
            },
            "test": {
                "path": str((output_dir / "test.jsonl").resolve()),
                "count": len(test_rows),
                "sha256": file_sha256(output_dir / "test.jsonl"),
            },
        },
        "source_counts": {
            "base_preserved": len(base_rows) - len(revisions),
            "base_revised": len(revisions),
            SOURCE: len(new_rows),
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("collect",), default="collect")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xlrs-json", type=Path)
    parser.add_argument("--mme-json", type=Path)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--fixed-test", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--target-count", type=int, default=4500)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = collect(
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        base_train=args.base_train,
        fixed_test=args.fixed_test,
        answer_index=load_answer_index(xlrs_json=args.xlrs_json, mme_json=args.mme_json),
        target_count=args.target_count,
        validate_only=args.validate_only,
        system_prompt=args.system_prompt,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "valid",
                    "validate_only",
                    "base_count",
                    "selected_new_count",
                    "accepted_new_count",
                    "revision_count",
                    "semantic_select",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
