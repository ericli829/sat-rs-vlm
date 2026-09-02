from __future__ import annotations

import json
import sys
from pathlib import Path

from taskgraph_lab import PROMPT_VERSION
from taskgraph_lab.generation.generate import RateLimiter, RuntimeSettings, process_sample
from taskgraph_lab.generation.provider import DryRunProvider
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl
from taskgraph_lab.taskgraph.enums import OperatorName
from taskgraph_lab.tests.dsl_fixtures import representative_graphs
from taskgraph_lab.tests.test_generation import REPAIR, REVIEW, SYSTEM, first_sample
from taskgraph_lab.tools.export_sft import main as export_sft_main


def test_every_canonical_operator_is_covered_by_round_trip_fixtures() -> None:
    covered = {
        OperatorName(node["op"])
        for graph in representative_graphs().values()
        for node in graph["nodes"]
    }
    assert covered == set(OperatorName)


def test_teacher_prompt_keeps_json_only_and_hides_student_grammar() -> None:
    assert PROMPT_VERSION == "taskgraph-v1.1-strict-batch-contract-v3"
    assert "Return JSON only." in SYSTEM
    assert "PLANNER SERIALIZATION NOTE" in SYSTEM
    assert "The Teacher must never emit, imitate, or manually translate that DSL." in SYSTEM
    for syntax in ("FINAL($", "FINAL_QUESTION(", "SELECT_REL(", 'T("'):
        assert syntax not in SYSTEM


def test_generation_attaches_compiler_derived_planner_dsl() -> None:
    outcome = process_sample(
        first_sample(),
        provider=DryRunProvider(),
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1, emit_planner_dsl=True),
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
        few_shot=None,
    )
    assert outcome.record is not None
    planner_dsl = outcome.record["planner_dsl"]
    assert canonicalize_target(parse_taskgraph_dsl(planner_dsl)) == outcome.record["target"]
    assert outcome.record["metadata"]["planner_dsl_version"]


def test_generation_can_explicitly_disable_planner_dsl() -> None:
    outcome = process_sample(
        first_sample(),
        provider=DryRunProvider(),
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1, emit_planner_dsl=False),
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
        few_shot=None,
    )
    assert outcome.record is not None
    assert "planner_dsl" not in outcome.record
    assert "planner_dsl_version" not in outcome.record["metadata"]


def test_sft_target_export_preserves_json_and_derived_dsl(tmp_path: Path, monkeypatch) -> None:
    graph = representative_graphs()["whole_image_count"]
    canonical = canonicalize_target(graph)
    record = {
        "sample_id": "fixture",
        "input": {"question": "How many?"},
        "target": canonical,
        "planner_dsl": compile_taskgraph_to_dsl(canonical),
        "metadata": {},
    }
    input_path = tmp_path / "accepted.jsonl"
    output_path = tmp_path / "sft.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_sft",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    assert export_sft_main() == 0
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["target"] == canonical
    assert exported["planner_dsl"] == record["planner_dsl"]
