from __future__ import annotations

import hashlib
import json
import logging

import pytest
from fastapi.testclient import TestClient

from sat_rs_vlm.competition.protocol_v2 import local_server
from sat_rs_vlm.competition.protocol_v2.image_package import ImagePackage
from sat_rs_vlm.competition.protocol_v2.local_server import LocalProtocolService
from sat_rs_vlm.competition.protocol_v2.server import create_app
from sat_rs_vlm.competition.protocol_v2.smoke import check_response, create_fixture_package, smoke_requests
from sat_rs_vlm.taskgraph.providers import FakeSemanticVLMProvider

SECRET = "local-test-only-token"
HEADERS = {"Authorization": f"Bearer {SECRET}"}


@pytest.fixture
def package(tmp_path):
    return ImagePackage(create_fixture_package(tmp_path / "sealed"))


def test_capture_is_private_opt_in_and_records_original_order(package, tmp_path, caplog):
    service = LocalProtocolService(package, mode="offline")
    target = tmp_path / "private" / "questions.jsonl"
    payload = smoke_requests(package)[0]
    payload["question"] = "\u524d\u540e\u65f6\u76f8\u6709\u53d8\u5316\u5417\uff1f\nSecond line"
    caplog.set_level(logging.INFO, logger="sat_rs_vlm.competition")
    with TestClient(create_app(service=service, api_key=SECRET, question_capture_path=target)) as client:
        assert client.post("/v1/predict", json=payload).status_code == 401
        assert client.post("/v1/predict", headers=HEADERS, json={"question": "invalid"}).status_code == 422
        assert target.read_text(encoding="utf-8") == ""
        response = client.post("/v1/predict", headers=HEADERS, json=payload)
        assert response.status_code == 200
        check_response(payload, response.json())
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["request"]["question"] == payload["question"]
    assert entry["request"]["images"] == payload["images"]
    assert entry["question_sha256"] == hashlib.sha256(payload["question"].encode()).hexdigest()
    assert SECRET not in lines[0]
    assert payload["question"] not in caplog.text
    assert "Second line" not in caplog.text
    assert "invalid" not in lines[0]


def test_capture_disabled_by_default(package):
    with TestClient(create_app(service=LocalProtocolService(package, mode="offline"), api_key=SECRET)) as client:
        assert client.app.state.question_capture is None
        assert client.post("/v1/predict", headers=HEADERS, json=smoke_requests(package)[0]).status_code == 200


def test_offline_accepts_all_contracts_with_unseen_item_ids(package):
    with TestClient(create_app(service=LocalProtocolService(package, mode="offline"), api_key=SECRET)) as client:
        for payload in smoke_requests(package):
            payload["item_id"] = "new-id-" + payload["response_constraint"]["type"]
            payload["question"] = "arbitrary question requiring no planner fixture"
            if payload["response_constraint"]["type"] == "integer":
                payload["response_constraint"]["minimum"] = 17
            response = client.post("/v1/predict", headers=HEADERS, json=payload)
            assert response.status_code == 200, response.text
            check_response(payload, response.json())
            if payload["response_constraint"]["type"] == "integer":
                assert response.json()["answer"] == "17"


class RecordingProvider(FakeSemanticVLMProvider):
    def release_idle_sessions(self):
        self.released = True


def test_2b_path_preserves_temporal_order_and_resolves_finite_ids(package):
    provider = RecordingProvider(default="urban")
    service = LocalProtocolService(package, mode="qwen2b", provider=provider)
    with TestClient(create_app(service=service, api_key=SECRET)) as client:
        for payload in smoke_requests(package)[:2]:
            response = client.post("/v1/predict", headers=HEADERS, json=payload)
            assert response.status_code == 200, response.text
            check_response(payload, response.json())
            if payload["response_constraint"]["type"] == "enum":
                model_input = provider.semantic_calls[-1].model_input
                assert model_input.visual_roles == ("BEFORE", "AFTER")
                assert "first image is BEFORE" in model_input.question
    assert provider.released


@pytest.mark.parametrize("text,status", [("7", 200), ("7 ships", 500), ("-1", 500)])
def test_2b_integer_is_strict_and_enforces_minimum(package, text, status):
    provider = RecordingProvider(default=text)
    service = LocalProtocolService(package, mode="qwen2b", provider=provider)
    payload = smoke_requests(package)[2]
    with TestClient(create_app(service=service, api_key=SECRET)) as client:
        response = client.post("/v1/predict", headers=HEADERS, json=payload)
        assert response.status_code == status
        if status == 200:
            assert response.json()["answer"] == "7"


def test_2b_normalized_bbox_is_converted_to_original_pixels(package):
    provider = RecordingProvider(default='{"bbox":[0.1,0.1,0.5,0.5]}')
    service = LocalProtocolService(package, mode="qwen2b", provider=provider)
    payload = smoke_requests(package)[4]
    with TestClient(create_app(service=service, api_key=SECRET)) as client:
        response = client.post("/v1/predict", headers=HEADERS, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["answer"] == [64, 48, 320, 240]
        wrong = {**payload, "image_width": 1}
        assert client.post("/v1/predict", headers=HEADERS, json=wrong).status_code == 422


@pytest.mark.parametrize("use_adapter", [False, True])
def test_local_factory_passes_optional_adapter_to_single_provider(package, tmp_path, monkeypatch, use_adapter):
    base = tmp_path / "base"
    base.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    captured = []

    def provider_factory(config, *, role):
        captured.append(config)
        return RecordingProvider(default="urban")

    config = tmp_path / "local-config.json"
    config.write_text(json.dumps({
        "mode": "qwen2b", "package": str(package.root), "preload": False,
        "model": {"model_id": str(base), "adapter_path": str(adapter) if use_adapter else None},
    }), encoding="utf-8")
    monkeypatch.setenv("COMPETITION_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("MODEL_API_KEY", SECRET)
    monkeypatch.setattr(local_server, "LazyQwenSemanticProvider", provider_factory)
    with TestClient(local_server.create_local_app()) as client:
        assert client.get("/healthz", headers=HEADERS).status_code == 200
    assert len(captured) == 1
    assert captured[0].adapter_path == (str(adapter) if use_adapter else None)


def test_local_factory_rejects_missing_explicit_adapter(package, tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    config = tmp_path / "local-config.json"
    config.write_text(json.dumps({
        "mode": "qwen2b", "package": str(package.root), "preload": False,
        "model": {"model_id": str(base), "adapter_path": str(tmp_path / "missing")},
    }), encoding="utf-8")
    monkeypatch.setenv("COMPETITION_LOCAL_CONFIG", str(config))
    with pytest.raises(ValueError, match="adapter directory"):
        local_server.create_local_app()
