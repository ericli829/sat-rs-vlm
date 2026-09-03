from __future__ import annotations

import json
from pathlib import Path

from sat_rs_vlm.evaluation.official_mcq import (
    parse_mme_realworld_choice,
    parse_reference_choices,
    parse_xlrs_choices,
)
from sat_rs_vlm.evaluation.protocols import load_contract, resolve_protocol
from sat_rs_vlm.evaluation.records import PredictionRecord
from sat_rs_vlm.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"


def _record(**kwargs: object) -> PredictionRecord:
    row: dict[str, object] = {
        "id": "sample",
        "task_type": "vqa",
        "prediction": "A",
        "reference": "A",
        "metadata": {},
    }
    row.update(kwargs)
    return PredictionRecord.from_mapping(row, 1)


def test_official_parsers_match_upstream_shapes() -> None:
    assert parse_mme_realworld_choice("The best answer is: (C)").choices == ("C",)
    assert parse_mme_realworld_choice(
        "blue", ["(A) red", "(B) blue"]
    ).choices == ("B",)
    assert parse_xlrs_choices("The answer is: (B) (A)").choices == ("A", "B")
    assert parse_reference_choices("A B", allowed=frozenset("ABCD"), single=False).parse_ok
    assert not parse_reference_choices("A B", allowed=frozenset("ABCD"), single=True).parse_ok


def test_official_protocol_routing_and_provenance() -> None:
    contract = load_contract(CONTRACT)
    assert resolve_protocol(
        _record(metadata={"dataset": "MME-RealWorld", "official_subtask": "Remote Sensing"}),
        contract,
    ).name == "mme_realworld_rs_mcq"
    xlrs = resolve_protocol(
        _record(
            metadata={
                "dataset": "XLRS-Bench",
                "official_category": "Land use classification/Overall Land use classification",
            }
        ),
        contract,
    )
    assert xlrs.name == "xlrs_vqa_multiselect"
    assert xlrs.metric_label == "official"
    assert xlrs.provenance["source_commit"] == "828ac0ae8f200f6b05ac9ab12554caee6078e336"


def test_official_mcq_evaluation_uses_exact_set_and_separate_groups(tmp_path: Path) -> None:
    source = tmp_path / "predictions.jsonl"
    rows = [
        {
            "id": "mme-1",
            "task_type": "vqa",
            "prediction": "The best answer is: C",
            "reference": "C",
            "metadata": {
                "dataset": "MME-RealWorld",
                "official_subtask": "Remote Sensing",
                "official_category": "Objects",
                "official_task": "Perception",
                "dataset_version": "official-2024",
                "split": "train",
                "language": "en",
                "prompt_profile": "mme_realworld_official_mcq_v1",
                "evaluation_scope": "official_full_split",
                "answer_choices": ["(A) no", "(B) no", "(C) yes", "(D) no", "(E) no"],
            },
        },
        {
            "id": "xlrs-1",
            "task_type": "vqa",
            "prediction": "A B",
            "reference": "A B",
            "metadata": {
                "dataset": "XLRS-Bench",
                "official_category": "Land use classification/Overall Land use classification",
                "official_subtask": "Overall Land use classification",
                "official_task": "Land use classification",
                "dataset_version": "XLRS-Bench-lite",
                "split": "train",
                "language": "en",
                "prompt_profile": "xlrs_bench_official_multiselect_v1",
                "evaluation_scope": "official_full_split",
            },
        },
        {
            "id": "xlrs-2",
            "task_type": "vqa",
            "prediction": "A",
            "reference": "A B",
            "metadata": {
                "dataset": "XLRS-Bench",
                "official_category": "Land use classification/Overall Land use classification",
                "official_subtask": "Overall Land use classification",
                "official_task": "Land use classification",
                "dataset_version": "XLRS-Bench-lite",
                "split": "train",
                "language": "en",
                "prompt_profile": "xlrs_bench_official_multiselect_v1",
                "evaluation_scope": "official_full_split",
            },
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    outputs = run_evaluation(
        source,
        tmp_path / "results",
        contract_path=CONTRACT,
        protected_repository=tmp_path / "protected",
        semantic_enabled=False,
    )
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    mme = summary["by_protocol"]["mme_realworld_rs_mcq"]
    assert mme["metrics"]["official_exact_accuracy"]["value"] == 1.0
    xlrs = summary["by_protocol"]["xlrs_vqa_multiselect"]
    assert xlrs["metrics"]["official_exact_accuracy"]["value"] == 0.5
    assert summary["protocol_provenance"]["mme_realworld_rs_mcq"]["metric_label"] == "official"
    assert (
        summary["official_comparability"]["xlrs_vqa_multiselect"]["status"]
        == "eligible_for_official_comparison"
    )
    evaluated = [
        json.loads(line)
        for line in outputs["evaluated_predictions"].read_text(encoding="utf-8").splitlines()
    ]
    assert evaluated[1]["sample_metrics"]["reference_choices"] == ["A", "B"]
    assert evaluated[2]["sample_metrics"]["exact_choice_match"] is False
