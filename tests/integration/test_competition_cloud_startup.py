"""Opt-in production assembly checks on the cloud host; weights must stay unloaded."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLOUD_STARTUP_TESTS") != "1",
    reason="requires configured cloud artifacts, but never loads model weights",
)
ROOT = Path(__file__).resolve().parents[2]


def test_production_factory_health_and_real_providers_without_loading_weights(monkeypatch):
    from sat_rs_vlm.competition.protocol_v2.server import create_app
    from sat_rs_vlm.integrations.counting import CountingSystemProvider
    from sat_rs_vlm.integrations.detectors.lae_dino_sidecar import _LAESidecarClient
    from sat_rs_vlm.integrations.retrievers.openclip import OpenCLIPRetrieverProvider
    from sat_rs_vlm.models.qwen3vl_shared import SharedQwen3VLModelManager
    from sat_rs_vlm.taskgraph.providers import (
        LazyQwenSemanticProvider,
        LocatorRegionRetrieverAdapter,
        ProposalDetectionAdapter,
        Qwen3VLPlannerProvider,
    )

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "this test must have no GPU visibility"
    config_path = Path(os.environ["COMPETITION_CONFIG"]).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config.get("offline") is False
    assert config.get("server", {}).get("preload", False) is False
    attempts = []

    def reject_load(*args, **kwargs):
        attempts.append("weight_or_sidecar_load")
        raise AssertionError("no-model startup attempted to load weights or start a model worker")

    for cls, method in (
        (SharedQwen3VLModelManager, "load"),
        (Qwen3VLPlannerProvider, "_load"),
        (LazyQwenSemanticProvider, "_load"),
        (OpenCLIPRetrieverProvider, "_load"),
        (_LAESidecarClient, "start"),
    ):
        monkeypatch.setattr(cls, method, reject_load)

    monkeypatch.chdir(ROOT)
    headers = {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}
    with TestClient(create_app(config_path)) as client:
        assert client.get("/healthz").status_code == 401
        response = client.get("/healthz", headers=headers)
        assert response.status_code == 200
        health = response.json()
        readiness = health["model_readiness"]
        assert health["status"] == "ready"
        assert readiness["preflight_errors"] == []
        assert readiness["models_loaded"] is False
        assert readiness["shared_4b_base"]["enabled"] is True
        assert readiness["shared_4b_base"]["loaded"] is False
        assert client.post("/v1/predict", headers=headers, json={}).status_code == 422
        assert client.post("/v1/predict", headers=headers, content="invalid").status_code == 415

        providers = client.app.state.service.runtime.providers
        assert isinstance(providers.planner, Qwen3VLPlannerProvider)
        assert isinstance(providers.semantic_2b, LazyQwenSemanticProvider)
        assert isinstance(providers.route_4b, LazyQwenSemanticProvider)
        assert providers.planner.shared_model_manager is providers.shared_model_manager
        assert providers.route_4b.shared_model_manager is providers.shared_model_manager
        assert isinstance(providers.detection._provider, ProposalDetectionAdapter)
        assert isinstance(providers.counting._provider, CountingSystemProvider)
        verifier = providers.detection._verifier
        assert verifier is providers.counting._verifier
        assert isinstance(verifier.retriever, OpenCLIPRetrieverProvider)
        assert verifier.retriever._model is None
        assert isinstance(providers.retriever, LocatorRegionRetrieverAdapter)
        assert isinstance(
            providers.retriever._locator.retriever_provider, OpenCLIPRetrieverProvider
        )
        assert providers.retriever._locator.retriever_provider._model is None
        assert providers.planner._model is None
        assert providers.semantic_2b._engine is None
        assert providers.route_4b._engine is None
        print("PRODUCTION_NO_MODEL_HEALTH=" + json.dumps(health, sort_keys=True))
    assert attempts == []


@pytest.mark.skipif(os.name == "nt", reason="Linux cloud launcher")
def test_cloud_launcher_uses_its_checkout_when_host_env_points_to_old_tree(tmp_path):
    script = ROOT / "scripts/competition/start_cloud_protocol_v2.sh"
    subprocess.run(["bash", "-n", str(script)], check=True, timeout=5)
    capture = tmp_path / "launch.json"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "Path(os.environ['LAUNCH_CAPTURE']).write_text(json.dumps({"
        "'cwd': os.getcwd(), 'argv': sys.argv[1:], "
        "'config': os.environ['COMPETITION_CONFIG'], "
        "'planner': os.environ['QWEN3VL_4B_PLANNER_LORA'], "
        "'counting_src': os.environ['COUNTING_SYSTEM_SRC'], "
        "'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "time.sleep(2)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    host_env = tmp_path / "host-env.sh"
    host_env.write_text(
        "PROJECT_ROOT=/missing/old-checkout\nCOMPETITION_CONFIG=/missing/old-config.yaml\n"
        "PYTHON=/missing/old-python\nQWEN3VL_4B_PLANNER_LORA=/missing/old-planner\n"
        "CUDA_VISIBLE_DEVICES=unexpected\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PROJECT_ROOT", None)
    env.pop("COMPETITION_CONFIG", None)
    env.update(
        AUTODL_ENV_FILE=str(host_env),
        PYTHON=str(fake_python),
        RUN_DIR=str(tmp_path / "service"),
        LAUNCH_CAPTURE=str(capture),
        BIND_HOST="127.0.0.1",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        ["bash", str(script), str(port)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    launched = json.loads(capture.read_text(encoding="utf-8"))
    assert Path(launched["cwd"]) == ROOT
    assert Path(launched["config"]) == ROOT / "configs/competition/protocol_v2.yaml"
    assert Path(launched["counting_src"]) == ROOT / "04_counting_system_plan/src"
    assert launched["cuda_visible"] == ""
    assert launched["planner"] == env["QWEN3VL_4B_PLANNER_LORA"]
    assert "sat_rs_vlm.competition.protocol_v2.server:create_app" in launched["argv"]
    assert launched["argv"][launched["argv"].index("--workers") + 1] == "1"
