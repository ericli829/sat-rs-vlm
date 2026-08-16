"""Evaluate the rule-plus-local-LLM hybrid against a frozen human gold set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.change_judge import (  # noqa: E402
    LOCAL_JUDGE_DECISION_PROFILE,
    LOCAL_JUDGE_IMPLEMENTATION_VERSION,
    conservative_rule_decision,
)

# Keep this historical field name so previously generated audit CSV files and
# downstream notebooks remain readable. The implementation version/profile in
# the summary are authoritative; as of this release they identify v2.3/v1.3.
LEGACY_HYBRID_DECISION_FIELD = "hybrid_v2_decision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-standard", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-status",
        choices=(
            "development_set_only",
            "independent_holdout_unadjudicated",
            "independent_holdout_final",
        ),
        default="development_set_only",
    )
    return parser.parse_args()


def _metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for row in rows:
        expected = row["human_gold_label"]
        predicted = row[field]
        if expected == 1 and predicted == 1:
            tp += 1
        elif expected == 0 and predicted == 0:
            tn += 1
        elif expected == 0:
            fp += 1
        else:
            fn += 1
    total = tp + tn + fp + fn
    recall_0 = tn / (tn + fp) if tn + fp else None
    recall_1 = tp / (tp + fn) if tp + fn else None
    return {
        "num_samples": total,
        "accuracy": (tp + tn) / total if total else None,
        "balanced_accuracy": (
            (recall_0 + recall_1) / 2 if recall_0 is not None and recall_1 is not None else None
        ),
        "change_precision": tp / (tp + fp) if tp + fp else None,
        "change_recall": recall_1,
        "change_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _paired(rows: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, Any]:
    both_correct = baseline_only = candidate_only = both_wrong = 0
    for row in rows:
        expected = row["human_gold_label"]
        baseline_correct = row[baseline] == expected
        candidate_correct = row[candidate] == expected
        if baseline_correct and candidate_correct:
            both_correct += 1
        elif baseline_correct:
            baseline_only += 1
        elif candidate_correct:
            candidate_only += 1
        else:
            both_wrong += 1
    total = len(rows)
    return {
        "num_samples": total,
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_wrong": both_wrong,
        "candidate_minus_baseline_accuracy": (
            (candidate_only - baseline_only) / total if total else None
        ),
    }


def evaluate_hybrid_rows(
    gold_rows: list[dict[str, str]],
    answer_rows: dict[str, dict[str, Any]],
    *,
    evidence_status: str = "development_set_only",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    for gold in gold_rows:
        audit_id = gold["audit_id"]
        label = gold["human_gold_label"]
        if label == "U":
            continue
        if label not in {"0", "1"} or audit_id not in answer_rows:
            raise ValueError(f"invalid gold row or missing answer key: {audit_id}")
        answer = answer_rows[audit_id]
        old = answer.get("old_parser_decision")
        local = answer.get("local_judge_decision")
        if old not in {0, 1} or local not in {0, 1}:
            raise ValueError(f"automatic decision is unresolved: {audit_id}")
        rule = conservative_rule_decision(gold["caption"])
        if rule is not None and rule.value in {0, 1}:
            hybrid = rule.value
            source = rule.source
            reason = rule.reason
        else:
            hybrid = local
            source = "local_llm_judge"
            reason = "fallback_to_existing_v3_judgment"
        evaluated.append(
            {
                "audit_id": audit_id,
                "caption": gold["caption"],
                "human_gold_label": int(label),
                "old_parser_decision": old,
                "local_v3_decision": local,
                LEGACY_HYBRID_DECISION_FIELD: hybrid,
                "hybrid_source": source,
                "hybrid_reason": reason,
                "hybrid_correct": hybrid == int(label),
            }
        )
    summary = {
        "schema_version": "1.0",
        "implementation_version": LOCAL_JUDGE_IMPLEMENTATION_VERSION,
        "decision_profile": LOCAL_JUDGE_DECISION_PROFILE,
        "compatibility": {
            "legacy_decision_field": LEGACY_HYBRID_DECISION_FIELD,
            "note": (
                "hybrid_v2 and hybrid_v2_decision are frozen schema names; "
                "implementation_version and decision_profile identify the active algorithm."
            ),
        },
        "evidence_status": evidence_status,
        "num_gold_binary_rows": len(evaluated),
        "source_distribution": dict(
            sorted(Counter(row["hybrid_source"] for row in evaluated).items())
        ),
        "old_contextual_parser": _metrics(evaluated, "old_parser_decision"),
        "local_llm_v3": _metrics(evaluated, "local_v3_decision"),
        "hybrid_v2": _metrics(evaluated, LEGACY_HYBRID_DECISION_FIELD),
        "hybrid_vs_old_paired": _paired(
            evaluated, "old_parser_decision", LEGACY_HYBRID_DECISION_FIELD
        ),
        "hybrid_vs_local_v3_paired": _paired(
            evaluated, "local_v3_decision", LEGACY_HYBRID_DECISION_FIELD
        ),
        "note": {
            "development_set_only": (
                "This set was used for error analysis. Do not claim generalization until an "
                "excluded blind holdout is labeled."
            ),
            "independent_holdout_unadjudicated": (
                "Independent excluded holdout using only annotator-agreement rows; final metrics "
                "require adjudication of remaining disagreements."
            ),
            "independent_holdout_final": (
                "Independent excluded holdout after double annotation and adjudication."
            ),
        }[evidence_status],
    }
    return summary, evaluated


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    with args.gold_standard.resolve().open(encoding="utf-8-sig", newline="") as file:
        gold_rows = list(csv.DictReader(file))
    answer_payload = json.loads(args.answer_key.resolve().read_text(encoding="utf-8-sig"))
    answer_rows = {str(row["audit_id"]): row for row in answer_payload["rows"]}
    try:
        summary, evaluated = evaluate_hybrid_rows(
            gold_rows,
            answer_rows,
            evidence_status=args.evidence_status,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(evaluated[0]) if evaluated else []
    with (output_dir / "hybrid_evaluated.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evaluated)
    print(f"Saved hybrid evaluation ({args.evidence_status}): {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
