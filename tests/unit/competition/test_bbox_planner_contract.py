"""CPU-only Protocol 2.0 / Planner DSL / unique BBOX final contract checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sat_rs_vlm.competition.protocol_v2.adapter import (
    CompetitionRequestAdapter,
    CompetitionRoutingConfig,
)
from sat_rs_vlm.competition.protocol_v2.errors import BBoxValidationError
from sat_rs_vlm.competition.protocol_v2.image_package import ImagePackage
from sat_rs_vlm.competition.protocol_v2.models import AnswerContract, parse_request
from sat_rs_vlm.competition.protocol_v2.server import create_app
from sat_rs_vlm.competition.protocol_v2.service import CompetitionService
from sat_rs_vlm.competition.protocol_v2.smoke import (
    check_response,
    create_fixture_package,
    smoke_requests,
)
from sat_rs_vlm.taskgraph.executor import ExecutorBinding
from sat_rs_vlm.taskgraph.operators import OperatorOutcome
from sat_rs_vlm.taskgraph.providers import Qwen3VLPlannerProvider
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    CandidateIntegrityError,
    Entity,
    EntitySet,
    ImageRef,
    Region,
    RegionSet,
    SelectResult,
    SelectStatus,
    unique_bbox_evidence,
)
from sat_rs_vlm.taskgraph.schema import OperatorName, TaskGraph


class DSLPlanner:
    provider_name = "mock_dsl_planner"

    def __init__(self, dsl: str):
        self.dsl = dsl
        self.calls = []
        self.user_payload = None

    def plan(self, request):
        self.calls.append(request)
        self.user_payload = json.loads(
            Qwen3VLPlannerProvider._messages(request, "mock-system")[1]["content"]
        )
        return Qwen3VLPlannerProvider._to_production_graph(self.dsl, request)

    def close(self):
        pass


SINGLE_DSL = 'INTENT(OTHER)\nn1=LOCATE($image0,T("ship"))\nFINAL($n1,BBOX)'


@pytest.fixture
def service(tmp_path):
    package = ImagePackage(create_fixture_package(tmp_path / "sealed"))
    config = yaml.safe_load(Path("configs/competition/runtime.offline.yaml").read_text())
    config["locate_refinement"] = {"single_detector_fallback": False}
    active = CompetitionService.from_config(package, config, offline=True)
    active.adapter = CompetitionRequestAdapter(CompetitionRoutingConfig(direct_vlm_max_pixels=1))
    active.runtime.providers.planner = DSLPlanner(SINGLE_DSL)
    active.runtime.providers.semantic_2b.responses["locate_single"] = json.dumps(
        {"status": "OK", "bbox": [64, 48, 320, 240]}
    )
    yield active
    active.close()


def bbox_request(service):
    return next(x for x in smoke_requests(service.package) if x["answer_type"] == "bbox")


@pytest.mark.parametrize(
    "dsl,model_box,expected",
    [
        (SINGLE_DSL, [64, 48, 320, 240], [64, 48, 320, 240]),
        (
            "INTENT(OTHER)\nn1=REGION($image0,BOTTOM_RIGHT)\n"
            'n2=LOCATE($n1,T("building",color="white"))\nFINAL($n2,BBOX)',
            [16, 24, 160, 120],
            [336, 262.5, 480, 352.5],
        ),
    ],
)
def test_bbox_request_dsl_to_unique_global_evidence(service, monkeypatch, dsl, model_box, expected):
    planner = DSLPlanner(dsl)
    service.runtime.providers.planner = planner
    semantic = service.runtime.providers.semantic_2b
    semantic.responses["locate_single"] = json.dumps({"status": "OK", "bbox": model_box})

    def forbidden(*args, **kwargs):
        raise AssertionError("BBOX must not generate choice text or detect a second target")

    monkeypatch.setattr(service.resolver.choice_resolver, "resolve", forbidden)
    monkeypatch.setattr(service.runtime.choice_resolver, "resolve", forbidden)
    monkeypatch.setattr(service.runtime.providers.detection, "detect", forbidden)
    payload = bbox_request(service)
    with TestClient(create_app(service=service, api_key="mock-only")) as client:
        response = client.post(
            "/v1/predict", headers={"Authorization": "Bearer mock-only"}, json=payload
        )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "protocol_version": "2.0",
        "request_id": payload["request_id"],
        "item_id": payload["item_id"],
        "answer": expected,
    }
    assert planner.user_payload == {
        "question": payload["question"]
        + "\nResolve one target and provide final evidence in original global pixel coordinates.",
        "question_type": "FREE_FORM",
        "choices": None,
        "answer_contract": {
            "kind": "bbox",
            "values": [],
            "case_insensitive": False,
            "minimum": None,
            "min_length": 0,
            "image_width": 640,
            "image_height": 480,
        },
        "expected_final_answer_type": "BBOX",
        "inputs": {
            "image0": {
                "type": "image",
                "uri_or_key": str(
                    service.package.resolve_all(parse_request(payload).images)[0].path
                ),
            }
        },
    }
    assert service.last_trace.taskgraph["final"] == {
        "sources": ["$n1" if dsl == SINGLE_DSL else "$n2"],
        "question": "",
        "answer_type": "BBOX",
    }
    locate = service.last_trace.nodes[-1]
    assert locate.trace_metadata["requirement"] == "SINGLE_TARGET"
    assert len(semantic.calls) == 1
    assert semantic.calls[0].output_contract == "locate_single"
    assert not semantic.choice_calls


@pytest.mark.parametrize(
    "output",
    [
        {"status": "NONE"},
        {"status": "AMBIGUOUS"},
        {"status": "UNRESOLVED"},
        {"status": "OK", "bbox": [-1, 0, 20, 30]},
        {"status": "OK", "bbox": [0, 0, 641, 480]},
        {"status": "OK", "bbox": [30, 20, 10, 40]},
        {"status": "OK", "bbox": [0, 0, float("nan"), 40]},
        {"status": "OK", "boxes": [[1, 2, 3, 4], [5, 6, 7, 8]]},
    ],
)
def test_invalid_single_locate_falls_back_to_original_image(service, output):
    service.runtime.providers.semantic_2b.responses["locate_single"] = json.dumps(output)
    with TestClient(create_app(service=service, api_key="mock-only")) as client:
        response = client.post(
            "/v1/predict",
            headers={"Authorization": "Bearer mock-only"},
            json=bbox_request(service),
        )
    assert response.status_code == 200
    assert response.json()["answer"] == [64, 48, 320, 240]
    assert service.last_diagnostics["fallback_triggered"] is True
    assert service.last_diagnostics["fallback_model_role"] == "semantic_2b"
    assert service.last_diagnostics["fallback_success"] is True
    assert not service.runtime.providers.semantic_2b.choice_calls


def test_invalid_single_locate_and_invalid_vlm_fallback_fail_closed(service):
    service.runtime.providers.semantic_2b.responses["locate_single"] = json.dumps(
        {"status": "UNRESOLVED"}
    )
    service.runtime.providers.semantic_2b.responses["bbox_json"] = "not valid bbox JSON"
    with TestClient(create_app(service=service, api_key="mock-only")) as client:
        response = client.post(
            "/v1/predict",
            headers={"Authorization": "Bearer mock-only"},
            json=bbox_request(service),
        )
    assert response.status_code == 500
    assert "answer" not in response.json()
    assert service.last_diagnostics["fallback_triggered"] is True
    assert service.last_diagnostics["fallback_success"] is False


@pytest.mark.parametrize("cardinality", [0, 2])
@pytest.mark.parametrize("deferred", [True, False])
def test_bbox_runtime_rejects_nonunique_provider_evidence(
    service, monkeypatch, cardinality, deferred
):
    class MockLocate:
        provider_name = "mock_locate"

        def execute(self, node, inputs, context):
            region = Region(inputs["image"], (10, 20, 30, 40))
            evidence = EntitySet(tuple(Entity(region, "ship") for _ in range(cardinality)))
            return OperatorOutcome(evidence, self.provider_name)

    def forbidden(*args, **kwargs):
        raise AssertionError("A nonunique BBOX final must not fall back to choice generation")

    service.runtime.graph_executor.router.bindings[OperatorName.LOCATE] = ExecutorBinding(
        MockLocate()
    )
    monkeypatch.setattr(service.runtime.choice_resolver, "resolve", forbidden)
    payload = parse_request(bbox_request(service))
    images = service.package.resolve_all(payload.images)
    request = service.adapter.adapt(payload, images, AnswerContract.from_request(payload))
    with pytest.raises(CandidateIntegrityError, match="exactly one target"):
        service.runtime.run(replace(request, defer_final_answer=deferred))


def test_bbox_runtime_enforces_uniqueness_even_when_not_deferred(service):
    payload = parse_request(bbox_request(service))
    images = service.package.resolve_all(payload.images)
    request = service.adapter.adapt(payload, images, AnswerContract.from_request(payload))
    result = service.runtime.run(replace(request, defer_final_answer=False))
    assert isinstance(result.output, Entity)
    assert result.output.region.bbox_xyxy_global == (64, 48, 320, 240)
    assert not service.runtime.providers.semantic_2b.choice_calls


@pytest.mark.parametrize("kind", ["bbox", "integer", "single_choice", "enum", "short_text"])
def test_all_five_http_answer_contracts_remain_compatible(service, kind):
    payload = next(
        x for x in smoke_requests(service.package) if x["response_constraint"]["type"] == kind
    )
    with TestClient(create_app(service=service, api_key="mock-only")) as client:
        response = client.post(
            "/v1/predict", headers={"Authorization": "Bearer mock-only"}, json=payload
        )
    assert response.status_code == 200, response.text
    check_response(payload, response.json())
    assert set(response.json()) == {"protocol_version", "request_id", "item_id", "answer"}


@pytest.mark.parametrize("collection", [EntitySet, RegionSet])
@pytest.mark.parametrize("cardinality", [0, 2])
def test_zero_and_multiple_bbox_targets_are_rejected(service, collection, cardinality, tmp_path):
    image = ImageRef(str(tmp_path / "visual"), 640, 480)
    region = Region(image, (10, 20, 30, 40))
    item = Entity(region, "ship") if collection is EntitySet else region
    evidence = collection(tuple(item for _ in range(cardinality)))
    with pytest.raises(CandidateIntegrityError):
        unique_bbox_evidence(evidence)
    with pytest.raises(BBoxValidationError):
        service.resolver.resolve(
            evidence, "Locate", AnswerContract("bbox", image_width=640, image_height=480), (image,)
        )


@pytest.mark.parametrize(
    "provenance",
    [
        {"resolution_status": "UNRESOLVED"},
        {"resolution_status": "AMBIGUOUS"},
        {"resolution_status": "PARTIAL"},
        {"candidate_pool_complete": False},
        {"candidate_truncated": True},
    ],
)
@pytest.mark.parametrize("level", ["selection", "collection", "entity", "region"])
def test_bbox_rejects_incomplete_or_unresolved_nested_evidence(
    service, tmp_path, provenance, level
):
    image = ImageRef(str(tmp_path / "visual"), 640, 480)
    region = Region(image, (10, 20, 30, 40), provenance if level == "region" else {})
    entity = Entity(region, "ship", provenance=provenance if level == "entity" else {})
    evidence = EntitySet((entity,), provenance if level == "collection" else {})
    if level == "selection":
        evidence = SelectResult(evidence, SelectStatus.OK, "mock", provenance=provenance)
    with pytest.raises(BBoxValidationError):
        service.resolver.resolve(
            evidence, "Locate", AnswerContract("bbox", image_width=640, image_height=480), (image,)
        )


@pytest.mark.parametrize("mismatch", ["path", "width", "height", "contract"])
def test_bbox_requires_matching_original_image_identity_and_dimensions(service, tmp_path, mismatch):
    image = ImageRef(str(tmp_path / "visual"), 640, 480)
    source_image = replace(
        image,
        **{
            "path": {"uri_or_key": str(tmp_path / "different")},
            "width": {"width": 320},
            "height": {"height": 240},
            "contract": {},
        }[mismatch],
    )
    contract = AnswerContract(
        "bbox", image_width=320 if mismatch == "contract" else 640, image_height=480
    )
    with pytest.raises(BBoxValidationError):
        service.resolver.resolve(
            Region(source_image, (10, 20, 30, 40)), "Locate", contract, (image,)
        )


@pytest.mark.parametrize("operator", ["GROUP", "COUNT", "VLM_REASON", "RELATION", "CLASSIFY"])
def test_production_schema_rejects_non_single_visual_bbox_sources(operator):
    graph = {
        "question": "Locate the ship.",
        "question_type": "FREE_FORM",
        "inputs": {"image0": {"uri_or_key": "synthetic:image0"}},
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship"}},
            },
        ],
        "final": {"sources": ["$n2"], "question": "", "answer_type": "BBOX"},
    }
    next_node = {
        "GROUP": {"op": "GROUP", "inputs": {"entities": "$n1"}, "params": {"mode": "CLUSTER"}},
        "COUNT": {
            "op": "COUNT",
            "inputs": {"entities": "$n1"},
            "params": {"target": {"category": "ship"}, "entire": False},
        },
        "VLM_REASON": {
            "op": "VLM_REASON",
            "inputs": {"image": "$image0"},
            "params": {"question": "Locate the ship."},
        },
        "RELATION": {
            "op": "RELATION",
            "inputs": {"subject": "$n1", "reference": "$n1"},
            "params": {},
        },
        "CLASSIFY": {
            "op": "CLASSIFY",
            "inputs": {"source": "$image0"},
            "params": {"label_space": ["ship"]},
        },
    }[operator]
    graph["nodes"].append({"id": "n2", **next_node})
    with pytest.raises(ValidationError, match="BBOX final requires"):
        TaskGraph.model_validate(graph)


def test_bbox_keeps_declared_json_answer_conversion_compatibility(service, tmp_path):
    image = ImageRef(str(tmp_path / "visual"), 640, 480)
    evidence = Answer(
        '{"bbox":[0.1,0.1,0.5,0.5]}', provenance={"coordinate_format": "normalized_0_1"}
    )
    result = service.resolver.resolve(
        evidence, "Locate", AnswerContract("bbox", image_width=640, image_height=480), (image,)
    )
    assert result == [64, 48, 320, 240]
