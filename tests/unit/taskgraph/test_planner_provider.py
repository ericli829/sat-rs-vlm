from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.providers import (
    ModelTaskGraphPlannerProvider,
    PlannerFailedError,
    PlannerRequest,
    Qwen3VLPlannerProvider,
)
from sat_rs_vlm.taskgraph.runtime import runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import ImageRef

ROOT = Path(__file__).resolve().parents[3]
DSL = """INTENT(SIMPLE_COUNT)
n1=COUNT($image0,T(\"ship\",size=\"large\"),true)
FINAL($n1,CHOICE_SINGLE)
"""


def _planner_dirs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "Qwen3-VL-4B-Instruct"
    adapter = tmp_path / "planner-lora"
    base.mkdir()
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen3-VL-4B-Instruct"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    return base, adapter


def _request() -> PlannerRequest:
    return PlannerRequest(
        question="How many large ships are there?",
        question_type="MULTIPLE_CHOICE_SINGLE",
        choices=("A 1", "B 2"),
        inputs={"$image0": ImageRef("fixture")},
        sample_id="sample-1",
    )


def _config(base: Path, adapter: Path) -> dict[str, object]:
    return {
        "providers": {
            "planner": {
                "kind": "qwen3vl_lora",
                "model_id": str(base),
                "adapter_path": str(adapter),
                "processor_id": str(base),
                "local_files_only": True,
            },
            "detection": {"kind": "fake"},
            "semantic_2b": {"kind": "fake"},
            "route_4b": {"kind": "fake"},
            "choice": {"reuse": "semantic_2b"},
            "region_retriever": {"kind": "fake"},
        },
        "capability_routing": {
            "ontology_path": str(ROOT / "configs/eval/semantic/remote_sensing_ontology.json")
        },
    }


def test_runtime_from_config_registers_qwen3vl_lora_planner(tmp_path: Path) -> None:
    base, adapter = _planner_dirs(tmp_path)
    runtime = runtime_from_config(_config(base, adapter))
    try:
        assert isinstance(runtime.providers.planner, Qwen3VLPlannerProvider)
        assert runtime.providers.planner.role == "planner_4b"
        assert runtime.providers.planner.provider_name == "qwen3vl_lora"
    finally:
        runtime.close()


def test_runtime_from_config_keeps_semantic_planner_compatibility(tmp_path: Path) -> None:
    base, adapter = _planner_dirs(tmp_path)
    config = _config(base, adapter)
    config["providers"]["planner"] = {"kind": "semantic", "provider": "route_4b"}
    runtime = runtime_from_config(config)
    try:
        assert isinstance(runtime.providers.planner, ModelTaskGraphPlannerProvider)
        assert runtime.providers.planner.provider is runtime.providers.route_4b
    finally:
        runtime.close()


def test_planner_missing_local_base_has_actionable_code(tmp_path: Path) -> None:
    adapter = tmp_path / "planner-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    with pytest.raises(FileNotFoundError, match="MISSING_LOCAL_PLANNER_MODEL"):
        Qwen3VLPlannerProvider(
            {
                "model_id": str(tmp_path / "missing-base"),
                "adapter_path": str(adapter),
                "local_files_only": True,
            }
        )


def test_planner_converts_lab_dsl_to_production_taskgraph() -> None:
    graph = Qwen3VLPlannerProvider._to_production_graph(DSL, _request())
    assert graph.question == "How many large ships are there?"
    assert graph.question_type.value == "MULTIPLE_CHOICE_SINGLE"
    assert graph.final.question == ""
    assert graph.nodes[0].op.value == "COUNT"
    assert graph.inputs["image0"].uri_or_key == "fixture"


def test_planner_retries_once_after_invalid_dsl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, adapter = _planner_dirs(tmp_path)
    provider = Qwen3VLPlannerProvider(
        {
            "model_id": str(base),
            "adapter_path": str(adapter),
            "local_files_only": True,
            "max_attempts": 2,
        }
    )
    outputs = iter(
        [
            ("not a dsl", {"termination_reason": "constraint_abort"}),
            (DSL, {"termination_reason": "final"}),
        ]
    )
    monkeypatch.setattr(provider, "_generate", lambda request, messages: next(outputs))

    graph = provider.plan(_request())

    assert graph.nodes[0].op.value == "COUNT"
    assert len(provider.last_metadata["attempts"]) == 2
    assert provider.last_metadata["status"] == "executed"
    assert provider.last_metadata["planner_output"] == DSL
    assert provider.last_metadata["attempts"][-1]["planner_output"] == DSL


def test_planner_failure_is_bounded_and_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, adapter = _planner_dirs(tmp_path)
    provider = Qwen3VLPlannerProvider(
        {
            "model_id": str(base),
            "adapter_path": str(adapter),
            "local_files_only": True,
            "max_attempts": 2,
        }
    )
    calls = 0

    def generate(
        request: PlannerRequest, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        return "not a dsl", {"termination_reason": "constraint_abort"}

    monkeypatch.setattr(provider, "_generate", generate)

    with pytest.raises(PlannerFailedError, match="planner_failed"):
        provider.plan(_request())
    assert calls == 2
    assert provider.last_metadata["status"] == "planner_failed"
