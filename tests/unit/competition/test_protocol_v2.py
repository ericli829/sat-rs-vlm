from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from sat_rs_vlm.competition.protocol_v2.adapter import CompetitionRequestAdapter
from sat_rs_vlm.competition.protocol_v2.answers import serialize_response, validate_bbox
from sat_rs_vlm.competition.protocol_v2.errors import (
    AnswerContractViolation,
    BBoxValidationError,
    ImageHashMismatchError,
    ImageNotFoundError,
    ProtocolValidationError,
    UnsupportedConstraintError,
)
from sat_rs_vlm.competition.protocol_v2.image_package import ImagePackage
from sat_rs_vlm.competition.protocol_v2.models import AnswerContract, ImageDescriptor, parse_request
from sat_rs_vlm.competition.protocol_v2.server import create_app
from sat_rs_vlm.competition.protocol_v2.service import (
    CompetitionService,
    validate_production_runtime,
)
from sat_rs_vlm.competition.protocol_v2.smoke import (
    check_response,
    create_fixture_package,
    smoke_requests,
)
from sat_rs_vlm.taskgraph.choice import ChoiceRequest
from sat_rs_vlm.taskgraph.providers import ChoiceScoringRequest, FixturePlannerProvider
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime_memory import RuntimeMemory, hash_question
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    Boolean,
    ChoiceScoreResult,
    Entity,
    ImageRef,
    Label,
    Region,
    ScalarInt,
)

SECRET = "unit-test-only-not-a-deployment-key"
HEADERS = {"Authorization": f"Bearer {SECRET}"}


@pytest.fixture
def package(tmp_path):
    return ImagePackage(create_fixture_package(tmp_path / "sealed"))


@pytest.fixture
def requests(package):
    return {p["response_constraint"]["type"]: p for p in smoke_requests(package)}


@pytest.fixture
def service(package):
    config = yaml.safe_load(Path("configs/competition/runtime.offline.yaml").read_text())
    active = CompetitionService.from_config(package, config, offline=True)
    yield active
    active.close()


@pytest.mark.parametrize("kind", ["enum", "single_choice", "integer", "short_text", "bbox"])
def test_parse_contracts(requests, kind):
    request = parse_request(requests[kind])
    assert AnswerContract.from_request(request).kind == kind


def test_online_joint_hash_overrides_adapter_route_using_raw_question(service, requests, tmp_path):
    request = parse_request(requests["short_text"])
    image = service.package.resolve_all(request.images)[0]
    path = tmp_path / "memory.jsonl"
    path.write_text(json.dumps({
        "sample_id": "historical-id", "mode": "DIRECT_VLM",
        "image_sha256": image.descriptor.sha256,
        "question_sha256": hash_question(request.question),
    }) + "\n", encoding="utf-8")
    service.runtime.runtime_memory = RuntimeMemory(path)
    service.predict(request)
    assert service.last_trace.telemetry["runtime_memory_hash_hit"] is True
    assert service.last_diagnostics["execution_mode"] == "DIRECT_VLM"


@pytest.mark.parametrize("kind", ["bbox", "integer"])
def test_adapter_preserves_raw_question_before_appending_instructions(package, requests, kind):
    payload = copy.deepcopy(requests[kind])
    if kind == "integer":
        payload["question"] = "How many ships are left of the airport?"
    request = parse_request(payload)
    images = package.resolve_all(request.images)
    adapted = CompetitionRequestAdapter().adapt(
        request, images, AnswerContract.from_request(request)
    )
    assert adapted.question != request.question
    assert adapted.memory_question == request.question
    assert adapted.verified_image_sha256s == tuple(image.descriptor.sha256 for image in images)


@pytest.mark.parametrize(
    "mutation",
    [
        {"protocol_version": "1.0"},
        {"images": []},
        {"question": " "},
        {"answer_type": "bbox"},
        {"unknown": "raw"},
        {"request_id": 123},
    ],
)
def test_invalid_requests(requests, mutation):
    payload = {**requests["enum"], **mutation}
    with pytest.raises(ProtocolValidationError):
        parse_request(payload)


def test_descriptor_validation_and_three_images(requests):
    for asset_id in ("../secret", "A" * 32, "a" * 31):
        payload = copy.deepcopy(requests["enum"])
        payload["images"][0]["asset_id"] = asset_id
        with pytest.raises(ProtocolValidationError):
            parse_request(payload)
    payload = copy.deepcopy(requests["enum"])
    payload["images"].append(payload["images"][0])
    with pytest.raises(ProtocolValidationError):
        parse_request(payload)


def test_constraint_validation(requests):
    for constraint in (
        {"type": "bbox"},
        {"type": "enum", "values": []},
        {"type": "enum", "values": ["Yes", "yes"], "case_insensitive": True},
        {"type": "integer", "minimum": True},
    ):
        with pytest.raises(ProtocolValidationError):
            parse_request({**requests["enum"], "response_constraint": constraint})
    with pytest.raises(UnsupportedConstraintError):
        parse_request({**requests["enum"], "response_constraint": {"type": "polygon"}})


def test_change_region(requests):
    request = parse_request(
        {
            **requests["enum"],
            "answer_type": "region_label",
            "response_constraint": {
                "type": "enum",
                "values": ["left", "right"],
                "question_form": "change_region",
            },
        }
    )
    assert AnswerContract.from_request(request).values == ("left", "right")


def test_arbitrary_choice_ids_order_and_alignment(service, requests):
    request = parse_request(requests["single_choice"])
    contract = AnswerContract.from_request(request)
    assert contract.values == ("opt-7", "Z", "B")
    assert contract.option_texts == ("urban", "forest", "water")
    resolver = service.runtime.choice_resolver
    result = resolver.resolve(
        ChoiceRequest(
            (Label("forest"),),
            "Which?",
            contract.option_texts,
            choice_ids=contract.values,
            strip_option_prefix=False,
        )
    )
    assert result.choice_id == "Z"
    with pytest.raises(ValueError):
        ChoiceRequest((Label("forest"),), "Which?", ("forest",), choice_ids=("Z", "B"))
    bad = copy.deepcopy(requests["single_choice"])
    bad["choices"].pop("B")
    with pytest.raises(ProtocolValidationError):
        parse_request(bad)


def test_legacy_benchmark_choices(service):
    request = ChoiceRequest((ScalarInt(7),), "Which?", ("A 5", "B 7", "C 9"))
    assert request.choice_ids == ("A", "B", "C")
    assert service.runtime.choice_resolver.resolve(request).choice_id == "B"


def test_precomputed_illegal_id_is_rejected(service):
    score = ChoiceScoreResult(
        selected_ids=("A",),
        scores={"A": 1},
        answer_type="CHOICE_SINGLE",
        reasoning_text=None,
        provider="offline",
        model_id="offline",
        method="test",
        cache_reused=True,
        latency_ms={},
        metadata={},
    )
    with pytest.raises(ValueError):
        service.runtime.choice_resolver.resolve(
            ChoiceRequest((score,), "Which?", ("urban", "water"), choice_ids=("Z", "B"))
        )


def test_cached_choice_preserves_ids_and_rejects_illegal(service, monkeypatch):
    provider = service.runtime.providers.choice
    request = ChoiceRequest(
        (ImageRef("tests/fixtures/miniature_dataset/images/vqa.ppm"),),
        "Which?",
        ("Urban", "Water"),
        choice_ids=("opt-7", "B"),
        strip_option_prefix=False,
    )
    assert service.runtime.choice_resolver.resolve(request).choice_id == "opt-7"
    assert provider.choice_calls[-1].choice_ids == ("opt-7", "B")
    assert json.loads(
        provider.choice_calls[-1].model_input.structured_context.split(
            "Choice ID mapping (JSON):\n"
        )[-1]
    ) == {"opt-7": "Urban", "B": "Water"}
    original = provider.reason_and_choose

    def illegal(call: ChoiceScoringRequest):
        return replace(original(call), selected_ids=("A",), scores={"A": 10})

    monkeypatch.setattr(provider, "reason_and_choose", illegal)
    with pytest.raises(ValueError):
        service.runtime.choice_resolver.resolve(request)


def test_enum_canonical_typed_and_kv_cache(service, requests):
    request = parse_request(requests["enum"])
    contract = AnswerContract.from_request(request)
    resolver = service.resolver
    assert resolver.resolve(Boolean(True), "Changed?", contract, ()) == "Yes"
    insensitive = replace(contract, case_insensitive=True)
    assert resolver.resolve(Label("yes"), "Changed?", insensitive, ()) == "Yes"
    images = tuple(ImageRef(str(x.path)) for x in service.package.resolve_all(request.images))
    assert resolver.resolve(images, "Changed?", contract, images) in contract.values
    assert service.runtime.providers.semantic_2b.semantic_calls[-1].candidate_ids == ("Yes", "No")
    assert resolver.decision_layer.config.legacy_text_fallback is False


@pytest.mark.parametrize("source", [Answer("3"), ScalarInt(True), ScalarInt(-1), Label("3")])
def test_integer_requires_authoritative_nonnegative_int(service, source):
    with pytest.raises(AnswerContractViolation):
        service.resolver.resolve(source, "Count?", AnswerContract("integer", minimum=0), ())


def test_integer_serialization_and_typed_short_text(service, requests):
    request = parse_request(requests["integer"])
    contract = AnswerContract.from_request(request)
    value = service.resolver.resolve(ScalarInt(3), "Count?", contract, ())
    assert type(value) is int
    assert serialize_response(request, contract, value)["answer"] == "3"
    provider = service.runtime.providers.semantic_2b
    for source, expected in (
        (Label("urban"), "urban"),
        (Boolean(False), "No"),
        (ScalarInt(3), "3"),
    ):
        assert (
            service.resolver.resolve(source, "Question?", AnswerContract("short_text"), ())
            == expected
        )
    assert provider.calls == []
    with pytest.raises(AnswerContractViolation):
        serialize_response(
            parse_request(requests["short_text"]), AnswerContract("short_text"), "x" * 4097
        )


@pytest.mark.parametrize(
    "text",
    [
        "<think>private reasoning</think> urban",
        "Analysis: private reasoning",
        "<analysis>private</analysis>",
    ],
)
def test_short_text_does_not_expose_marked_reasoning(requests, text):
    request = parse_request(requests["short_text"])
    with pytest.raises(AnswerContractViolation):
        serialize_response(request, AnswerContract.from_request(request), text)


@pytest.mark.parametrize(
    "bbox",
    [
        [-1, 0, 10, 10],
        [0, 0, 641, 480],
        [10, 0, 1, 10],
        [0, 0, 1, 0],
        [0, 0, float("nan"), 3],
        [0, 0, float("inf"), 3],
        [0, True, 3, 4],
        "0,0,3,4",
    ],
)
def test_bbox_validation(bbox):
    with pytest.raises(BBoxValidationError):
        validate_bbox(bbox, AnswerContract("bbox", image_width=640, image_height=480))


def test_bbox_global_entity_and_normalized_evidence(service):
    asset = next(iter(service.package.assets.values()))
    resolved = service.package.resolve(
        ImageDescriptor.model_validate(asset.model_dump(exclude={"size"}))
    )
    image = ImageRef(str(resolved.path), resolved.width, resolved.height)
    contract = AnswerContract("bbox", image_width=640, image_height=480)
    region = Region(image, (64, 48, 320, 240))
    assert service.resolver.resolve(Entity(region, "ship"), "Locate", contract, (image,)) == [
        64,
        48,
        320,
        240,
    ]
    for provenance in (
        {"normalized": True},
        {"coordinate_space": "crop_local"},
    ):
        with pytest.raises(BBoxValidationError):
            service.resolver.resolve(
                replace(region, provenance=provenance), "Locate", contract, (image,)
            )
    assert service.resolver.resolve(
        replace(region, provenance={"coordinate_space": "normalized_1000"}),
        "Locate",
        contract,
        (image,),
    ) == pytest.approx([40.96, 23.04, 204.8, 115.2])
    with pytest.raises(BBoxValidationError):
        service.resolver.resolve([0.1, 0.1, 0.5, 0.5], "Locate", contract, (image,))


def test_manifest_dynamic_dataset_order_and_hash_cache(package, requests, monkeypatch):
    payload = parse_request(requests["enum"])
    resolved = package.resolve_all(list(reversed(payload.images)))
    assert [x.descriptor.asset_id for x in resolved] == [
        x.asset_id for x in reversed(payload.images)
    ]
    assert all(x.path.suffix == "" for x in resolved)
    import sat_rs_vlm.competition.protocol_v2.image_package as module

    def forbidden_hash():
        raise AssertionError("unchanged file must reuse its validated hash")

    monkeypatch.setattr(module.hashlib, "sha256", forbidden_hash)
    package.resolve_all(payload.images)


def test_manifest_tampering_missing_and_mismatch(package, requests):
    image = parse_request(requests["enum"]).images[0]
    with pytest.raises(ImageHashMismatchError):
        package.resolve(image.model_copy(update={"sha256": "f" * 64}))
    with pytest.raises(ImageNotFoundError):
        package.resolve(image.model_copy(update={"asset_id": "f" * 32}))
    path = package.resolve(image).path
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ImageHashMismatchError):
        package.resolve(image)
    path.unlink()
    with pytest.raises(ImageNotFoundError):
        package.resolve(image)


def test_invalid_manifest_and_symlink_escape(tmp_path, package, requests):
    root = tmp_path / "invalid"
    root.mkdir()
    (root / "manifest.json").write_text('{"format":"bad","dataset_id":"dynamic","assets":[]}')
    with pytest.raises(ProtocolValidationError):
        ImagePackage(root)
    asset = parse_request(requests["enum"]).images[0]
    path = package.root / "assets" / asset.asset_id
    outside = tmp_path / "outside.png"
    path.replace(outside)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlinks")
    with pytest.raises(ImageNotFoundError):
        package.resolve(asset)


def test_competition_routing_uses_semantics_not_dimensions(package, requests):
    adapter = CompetitionRequestAdapter()
    for kind, payload in requests.items():
        request = parse_request(payload)
        adapted = adapter.adapt(
            request, package.resolve_all(request.images), AnswerContract.from_request(request)
        )
        expected = {
            "integer": ExecutionMode.DIRECT_DETECTION,
            "bbox": ExecutionMode.TASKGRAPH_UHR,
        }.get(kind, ExecutionMode.DIRECT_VLM)
        assert adapted.execution_mode == expected
        assert adapted.defer_final_answer
        assert adapted.dataset == "competition_protocol_v2"
        if kind == "enum":
            assert adapted.image_paths[0].endswith(request.images[0].asset_id)
            assert adapted.task_category == "temporal_change"
    request = parse_request(
        {**requests["integer"], "question": "How many ships are left of the port?"}
    )
    assert (
        adapter.adapt(
            request, package.resolve_all(request.images), AnswerContract.from_request(request)
        ).execution_mode
        == ExecutionMode.TASKGRAPH_UHR
    )
    request = parse_request(requests["short_text"])
    images = tuple(
        replace(x, width=16000, height=16000) for x in package.resolve_all(request.images)
    )
    assert (
        adapter.adapt(request, images, AnswerContract.from_request(request)).execution_mode
        == ExecutionMode.DIRECT_VLM
    )


def test_production_never_defaults_to_fake():
    offline = yaml.safe_load(Path("configs/competition/runtime.offline.yaml").read_text())
    for config in ({}, offline):
        with pytest.raises(ValueError):
            validate_production_runtime(config)
    real = yaml.safe_load(
        Path("configs/taskgraph/runtime.real.example.yaml").read_text(encoding="utf-8")
    )
    validate_production_runtime(real)


def test_http_auth_echo_five_contracts_and_safe_logs(service, requests, caplog):
    with TestClient(create_app(service=service, api_key=SECRET)) as client:
        for headers in ({}, {"Authorization": "Bearer bad"}):
            assert client.get("/healthz", headers=headers).status_code == 401
            assert (
                client.post("/v1/predict", headers=headers, json=requests["enum"]).status_code
                == 401
            )
        health = client.get("/healthz", headers=HEADERS)
        assert health.json() == {
            "status": "ready",
            "protocol_version": "2.0",
            "dataset_id": service.package.dataset_id,
        }
        caplog.set_level(logging.INFO, logger="sat_rs_vlm.competition")
        for kind, payload in requests.items():
            response = client.post("/v1/predict", headers=HEADERS, json=payload)
            assert response.status_code == 200, (kind, response.text)
            assert response.headers["content-type"] == "application/json"
            check_response(payload, response.json())
            if kind == "bbox":
                assert response.json()["answer"] == [64, 48, 320, 240]
        assert SECRET not in caplog.text
        assert requests["short_text"]["question"] not in caplog.text
        assert str(service.package.root) not in caplog.text
        assert client.post("/infer", headers=HEADERS, json={}).status_code == 404
        assert client.get("/healthz/", headers=HEADERS).status_code == 404


@pytest.mark.parametrize(
    "box,metadata",
    [
        ([0.1, 0.1, 0.5, 0.5], {}),
        ([100, 100, 500, 500], {"coordinate_format": "scaled_0_1000"}),
        ([64, 48, 320, 240], {"coordinate_format": "pixel_xyxy"}),
    ],
)
def test_http_model_bbox_conversion_and_private_trace(
    service, requests, monkeypatch, box, metadata
):
    active = CompetitionService(
        service.package,
        service.runtime,
        {"bbox": {"model_output_coordinate_format": "normalized_0_1"}},
    )
    original_run = active.runtime.run

    def model_bbox(request):
        result = original_run(request)
        return replace(
            result,
            output=Answer(json.dumps({"label": "ship", "bbox": box}), provenance=metadata),
        )

    monkeypatch.setattr(active.runtime, "run", model_bbox)
    with TestClient(create_app(service=active, api_key=SECRET)) as client:
        response = client.post("/v1/predict", headers=HEADERS, json=requests["bbox"])
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["answer"] == [64, 48, 320, 240]
        assert set(payload) == {"protocol_version", "request_id", "item_id", "answer"}
        conversion = active.last_trace.telemetry["competition_bbox_conversion"]
        assert conversion["output_bbox_xyxy_global"] == [64, 48, 320, 240]
        assert conversion["converted"] is (metadata.get("coordinate_format") != "pixel_xyxy")


def test_http_sanitized_errors_and_content_type(service, requests, monkeypatch):
    with TestClient(create_app(service=service, api_key=SECRET)) as client:
        assert client.post("/v1/predict", headers=HEADERS, content="oops").status_code == 415
        response = client.post(
            "/v1/predict", headers=HEADERS, json={"question": "private question"}
        )
        assert response.status_code == 422
        assert "private question" not in response.text
        mismatch = {**requests["bbox"], "image_width": 1}
        assert client.post("/v1/predict", headers=HEADERS, json=mismatch).status_code == 422

        def broken(request):
            raise RuntimeError("secret full path and traceback")

        monkeypatch.setattr(service, "predict", broken)
        response = client.post("/v1/predict", headers=HEADERS, json=requests["enum"])
        assert response.status_code == 500
        assert "secret" not in response.text


def test_config_factory_smoke(package, requests, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", SECRET)
    monkeypatch.setenv("COMPETITION_IMAGE_PACKAGE", str(package.root))
    with TestClient(create_app("configs/competition/protocol_v2.offline.yaml")) as client:
        for payload in requests.values():
            response = client.post("/v1/predict", headers=HEADERS, json=payload)
            assert response.status_code == 200, response.text
            check_response(payload, response.json())


def test_timeout_keeps_gpu_serialized(service, requests, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = service.predict

    def slow(request):
        started.set()
        release.wait(2)
        try:
            return original(request)
        finally:
            finished.set()

    monkeypatch.setattr(service, "predict", slow)
    app = create_app(service=service, api_key=SECRET, timeout_seconds=0.01)

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/v1/predict", headers=HEADERS, json=requests["enum"])
                assert started.is_set()
                assert response.status_code == 504
                assert app.state.lock.locked()
                assert (
                    await client.post("/v1/predict", headers=HEADERS, json=requests["enum"])
                ).status_code == 503
                release.set()
                await app.state.inflight
                assert finished.is_set()
                assert not app.state.lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize("mime,format", [("image/jpeg", "JPEG"), ("image/webp", "WEBP")])
def test_extensionless_other_formats(tmp_path, mime, format):
    import hashlib

    root = tmp_path / "mime"
    (root / "assets").mkdir(parents=True)
    path = root / "assets" / ("a" * 32)
    Image.new("RGB", (64, 48), "green").save(path, format=format)
    asset = {
        "asset_id": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mime_type": mime,
        "size": path.stat().st_size,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "rsu-sealed-images-v1",
                "dataset_id": "dynamic-mime-package",
                "assets": [asset],
            }
        )
    )
    package = ImagePackage(root)
    descriptor = ImageDescriptor.model_validate({k: v for k, v in asset.items() if k != "size"})
    resolved = package.resolve(descriptor)
    assert (resolved.width, resolved.height) == (64, 48)
    assert resolved.path.suffix == ""
    with pytest.raises(ImageHashMismatchError):
        package.resolve(descriptor.model_copy(update={"mime_type": "image/png"}))


def test_dynamic_dataset_id_from_manifest(package, monkeypatch):
    path = package.root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dataset_id"] = "an-unseen-formal-package"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("COMPETITION_IMAGE_PACKAGE", str(package.root))
    monkeypatch.setenv("MODEL_API_KEY", SECRET)
    with TestClient(create_app("configs/competition/protocol_v2.offline.yaml")) as client:
        assert (
            client.get("/healthz", headers=HEADERS).json()["dataset_id"] == manifest["dataset_id"]
        )


def test_two_image_bbox_selects_global_image_evidence(service, requests):
    payload = {**requests["bbox"], "images": requests["enum"]["images"]}
    request = parse_request(payload)
    contract = AnswerContract.from_request(request)
    adapted = service.adapter.adapt(request, service.package.resolve_all(request.images), contract)
    graph = json.loads(Path("tests/fixtures/competition/planner.json").read_text())["smoke-bbox"]
    graph["question"] = adapted.question
    graph["inputs"]["image1"] = {"type": "image", "uri_or_key": "dataset:image1"}
    graph["nodes"][0]["inputs"]["image"] = "$image1"
    service.runtime.providers.planner = FixturePlannerProvider({"smoke-bbox": graph})
    response = service.predict(request)
    check_response(payload, response)
    assert response["answer"] == [64, 48, 320, 240]


def test_graph_integer_returns_scalar_without_reasking(service, requests):
    payload = {**requests["integer"], "question": "How many ships are left of the port?"}
    request = parse_request(payload)
    contract = AnswerContract.from_request(request)
    adapted = service.adapter.adapt(request, service.package.resolve_all(request.images), contract)
    graph = {
        "version": "taskgraph-v1.1",
        "question": adapted.question,
        "question_type": "INTEGER",
        "inputs": {"image0": {"type": "image", "uri_or_key": "image"}},
        "intent": "RELATIONAL_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship"}, "entire": True},
            }
        ],
        "final": {"sources": ["$n1"], "question": "", "answer_type": "INTEGER"},
    }
    service.runtime.providers.planner = FixturePlannerProvider({"smoke-integer": graph})
    response = service.predict(request)
    assert response["answer"] == "1"
    assert service.runtime.providers.semantic_2b.calls == []
    assert service.runtime.providers.semantic_2b.choice_calls == []


@pytest.mark.parametrize(
    "change", [{"cache_reused": False}, {"selected_ids": ("illegal",), "scores": {"illegal": 1}}]
)
def test_enum_rejects_uncached_or_illegal_decision(service, requests, monkeypatch, change):
    provider = service.runtime.providers.semantic_2b
    original = provider.reason_and_decide
    monkeypatch.setattr(provider, "reason_and_decide", lambda req: replace(original(req), **change))
    with TestClient(create_app(service=service, api_key=SECRET)) as client:
        response = client.post("/v1/predict", headers=HEADERS, json=requests["enum"])
        assert response.status_code == 500
