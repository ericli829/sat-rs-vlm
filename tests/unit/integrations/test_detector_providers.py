from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from scripts.integrations import lae_dino_worker
from scripts.integrations import precompute_vlm_fo1_proposals as precompute
from scripts.integrations.lae_dino_worker import (
    _extract_predictions,
    _load_detector,
    _patch_lae_config_for_local_runtime,
    _validate_local_bert_root,
)
from scripts.integrations.vlm_fo1_worker import validate_request

from sat_rs_vlm.integrations.detectors import protocol as detector_protocol
from sat_rs_vlm.integrations.detectors.cache import ProposalCache
from sat_rs_vlm.integrations.detectors.config import expand_config_value, resolve_config_path
from sat_rs_vlm.integrations.detectors.grounding_dino import GroundingDinoProvider, _nms
from sat_rs_vlm.integrations.detectors.lae_dino_sidecar import (
    SidecarProtocolError,
    _LAESidecarClient,
)
from sat_rs_vlm.integrations.detectors.protocol import (
    ProposalError,
    ProposalResult,
    canonicalize_proposals,
    proposal_cache_key,
)
from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider
from sat_rs_vlm.integrations.vlm_fo1 import request_has_reference_leak


def test_provider_registry_is_lazy_and_exposes_mock() -> None:
    provider = create_proposal_provider("mock", {})
    try:
        assert provider.provider_name == "mock"
    finally:
        provider.close()
    with pytest.raises(ProposalError, match="unsupported proposal provider"):
        create_proposal_provider("unknown", {})


def test_detector_config_environment_expansion_never_leaves_literal(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "grounding"
    model_dir.mkdir()
    monkeypatch.setenv("TEST_DETECTOR_MODEL", str(model_dir))
    normalized = expand_config_value({"model_path": "${TEST_DETECTOR_MODEL}"})
    assert normalized["model_path"] == str(model_dir)
    assert "${" not in str(resolve_config_path(normalized["model_path"], label="model"))
    monkeypatch.delenv("TEST_DETECTOR_MODEL")
    with pytest.raises(ProposalError, match="unresolved detector configuration"):
        expand_config_value({"model_path": "${TEST_DETECTOR_MODEL}"})


def test_reference_guard_covers_provider_metadata() -> None:
    assert request_has_reference_leak({"proposal_metadata": {"reference": "secret"}})


def test_precomputed_proposal_failure_is_not_silently_counted_as_zero() -> None:
    normalized, response = validate_request(
        {
            "id": "failed",
            "image": "image.png",
            "question": "How many airplanes are visible?",
            "target_phrase": "airplanes",
            "bbox_list": [],
            "bbox_scores": [],
            "proposal_metadata": {"status": "failed", "error": "detector crashed"},
        },
        prompt_profile="official_fo1",
        proposal_backend="precomputed",
    )
    assert normalized is None
    assert response is not None
    assert response["failure_stage"] == "proposal_generation"


def test_canonicalize_normalized_invalid_and_deterministic_order() -> None:
    boxes, scores, stats = canonicalize_proposals(
        [[0.8, 0.8, 0.2, 0.2], [0.0, 0.0, 1.2, 1.0], [1.0, 1.0, 1.0, 1.0], [float("nan")] * 4],
        [0.5, 0.9, 0.99, 0.8],
        image_width=100,
        image_height=50,
        coordinate_mode="normalized",
        top_k=2,
    )
    assert boxes == [[0.0, 0.0, 100.0, 50.0], [20.0, 10.0, 80.0, 40.0]]
    assert scores == [0.9, 0.5]
    assert stats == {"invalid_count": 2, "reordered_count": 2, "clamped_count": 1}


def test_canonicalize_rejects_mismatched_lengths() -> None:
    assert canonicalize_proposals([], [], image_width=10, image_height=10) == (
        [],
        [],
        {"invalid_count": 0, "reordered_count": 0, "clamped_count": 0},
    )
    with pytest.raises(ProposalError, match="length mismatch"):
        canonicalize_proposals([[0, 0, 1, 1]], [], image_width=10, image_height=10)


def test_optional_nms_is_deterministic_and_empty_safe() -> None:
    boxes, scores = _nms(
        [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
        [0.8, 0.7, 0.6],
        0.5,
    )
    assert boxes == [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert scores == [0.8, 0.6]
    assert _nms([], [], 0.5) == ([], [])


def test_grounding_dino_uses_structured_text_labels_without_loading_model(tmp_path: Path) -> None:
    from PIL import Image

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    image_path = tmp_path / "image.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)
    provider = GroundingDinoProvider({"model_path": str(model_dir), "device": "cpu"})

    class FakeTorch:
        class _Inference:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def inference_mode(self):
            return self._Inference()

    class FakeProcessor:
        def __call__(self, *, images, text, return_tensors):
            self.call_text = text
            return {"input_ids": "input-ids", "pixel_values": "pixels"}

        def post_process_grounded_object_detection(self, outputs, *, input_ids, threshold,
                                                   text_threshold, target_sizes, text_labels):
            self.post_text_labels = text_labels
            self.post_input_ids = input_ids
            return [{"boxes": [], "scores": []}]

    class FakeModel:
        def __call__(self, **_kwargs):
            return object()

    processor = FakeProcessor()
    provider._processor = processor
    provider._model = FakeModel()
    provider._torch = FakeTorch()
    result = provider.predict(image_path, "Airplanes")
    assert processor.call_text == [["airplanes"]]
    assert processor.post_text_labels == [["airplanes"]]
    assert processor.post_input_ids == "input-ids"
    assert result.boxes_xyxy == []


def test_cache_round_trip_and_key_includes_query_and_parameters(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    key = proposal_cache_key(
        provider="mock",
        model_identity={"revision": "v1"},
        image=image,
        target_phrase="airplanes",
        parameters={"top_k": 10},
    )
    assert key != proposal_cache_key(
        provider="mock",
        model_identity={"revision": "v1"},
        image=image,
        target_phrase="ships",
        parameters={"top_k": 10},
    )
    cache = ProposalCache(tmp_path / "cache")
    result = ProposalResult(
        boxes_xyxy=[[1, 2, 3, 4]],
        scores=[0.75],
        latency_ms=1,
        provider="mock",
        model_id="mock-v1",
    )
    assert cache.get(key) is None
    cache.put(key, result)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.to_dict() == result.to_dict()
    assert cache.misses == 1
    assert cache.hits == 1


def test_lae_cache_identity_changes_with_config_bert_and_source_revision(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.py"
    config.write_text("model = 1\n", encoding="utf-8")
    bert = tmp_path / "bert"
    bert.mkdir()
    (bert / "config.json").write_text("{}", encoding="utf-8")
    base = {
        "checkpoint": str(checkpoint),
        "config_path": str(config),
        "bert_root": str(bert),
        "source_revision": "rev-a",
        "inference_query_mode": "target_conditioned_text_prompt",
    }
    first = precompute._manifest_model_identity("lae_dino_lae1m", base)
    config.write_text("model = 2\n", encoding="utf-8")
    config_changed = precompute._manifest_model_identity("lae_dino_lae1m", base)
    (bert / "config.json").write_text('{"changed":true}', encoding="utf-8")
    bert_changed = precompute._manifest_model_identity("lae_dino_lae1m", base)
    revision_changed = precompute._manifest_model_identity(
        "lae_dino_lae1m", {**base, "source_revision": "rev-b"}
    )
    assert first != config_changed
    assert config_changed != bert_changed
    assert bert_changed != revision_changed


def test_precompute_uses_target_phrase_and_never_sends_reference(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    rows = [
        {
            "id": "sample-1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.name},
                        {"type": "text", "text": "How many airplanes are visible?"},
                    ],
                },
                {"role": "assistant", "content": "999 airplanes; secret reference answer"},
            ],
        }
    ]
    provider = create_proposal_provider("mock", {})
    monkeypatch.setattr(precompute, "create_proposal_provider", lambda *_args, **_kwargs: provider)
    output, counts = precompute.precompute_rows(
        rows,
        provider_name="mock",
        provider_config={},
        image_root=tmp_path,
    )
    assert counts == {"ok": 1, "unsupported": 0, "failed": 0}
    assert provider.calls == [(image.resolve(), "airplanes")]
    assert output[0]["target_phrase"] == "airplanes"
    assert output[0]["proposal_metadata"]["target_phrase"] == "airplanes"
    # The original row remains intact for evaluator-side metrics, but its
    # assistant/reference content was not present in the provider request.
    assert "secret reference answer" in output[0]["messages"][1]["content"]


def test_precompute_row_failure_does_not_stop_following_rows(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")

    class FlakyProvider:
        provider_name = "mock"
        calls = 0

        def predict(self, _image_path: Path, _target_phrase: str) -> ProposalResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("one sample failed")
            return ProposalResult([[0, 0, 1, 1]], [0.5], 1.0, "mock", "flaky")

        def close(self) -> None:
            return None

    provider = FlakyProvider()
    monkeypatch.setattr(precompute, "create_proposal_provider", lambda *_args, **_kwargs: provider)
    rows = [
        {"id": "one", "question": "How many airplanes are visible?", "image": image.name},
        {"id": "two", "question": "How many airplanes are visible?", "image": image.name},
    ]
    output, counts = precompute.precompute_rows(
        rows, provider_name="mock", provider_config={}, image_root=tmp_path
    )
    assert counts == {"ok": 1, "unsupported": 0, "failed": 1}
    assert output[0]["proposal_metadata"]["status"] == "failed"
    assert output[1]["proposal_metadata"]["status"] == "ok"


def _sidecar_fixture(tmp_path: Path, body: str) -> tuple[_LAESidecarClient, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = tmp_path / "config.py"
    config.write_text("# explicit test config\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    bert_root = tmp_path / "bert"
    bert_root.mkdir()
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    json.loads(line)\n"
        f"    sys.stdout.write({body!r} + '\\n'); sys.stdout.flush()\n",
        encoding="utf-8",
    )
    client = _LAESidecarClient(
        {
            "source_root": str(source_root),
            "config_path": str(config),
            "checkpoint": str(checkpoint),
            "bert_root": str(bert_root),
            "worker_python": sys.executable,
            "worker_script": str(worker),
            "device": "cpu",
        }
    )
    return client, image


def test_lae_sidecar_rejects_malformed_json(tmp_path: Path) -> None:
    client, image = _sidecar_fixture(tmp_path, "'not-json'")
    try:
        with pytest.raises(ProposalError, match="invalid JSON"):
            client.request(image, "airplanes")
    finally:
        client.close()


def test_lae_sidecar_rejects_response_id_mismatch(tmp_path: Path) -> None:
    body = '{"id":"wrong-id","status":"ok","bbox_list":[],"bbox_scores":[]}'
    client, image = _sidecar_fixture(tmp_path, body)
    try:
        with pytest.raises(SidecarProtocolError, match="id mismatch") as error:
            client.request(image, "airplanes")
        assert error.value.failure_stage == "worker_protocol"
    finally:
        client.close()


def test_lae_sidecar_reports_process_crash(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = tmp_path / "config.py"
    config.write_text("# explicit test config\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    bert_root = tmp_path / "bert"
    bert_root.mkdir()
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    worker = tmp_path / "crash.py"
    worker.write_text("raise SystemExit(7)\n", encoding="utf-8")
    client = _LAESidecarClient(
        {
            "source_root": str(source_root),
            "config_path": str(config),
            "checkpoint": str(checkpoint),
            "bert_root": str(bert_root),
            "worker_python": sys.executable,
            "worker_script": str(worker),
        }
    )
    try:
        with pytest.raises(ProposalError, match="exited"):
            client.request(image, "airplanes")
    finally:
        client.close()


def test_lae_worker_extracts_mmdetection2_boxes_without_segmentation_payload() -> None:
    boxes, scores = _extract_predictions(
        (
            [[[1.0, 2.0, 10.0, 20.0, 0.8]]],
            [[[0, 1], [1, 0]]],
        )
    )
    assert boxes == [[1.0, 2.0, 10.0, 20.0]]
    assert scores == [0.8]


def test_lae_worker_requires_complete_local_bert_assets(tmp_path: Path) -> None:
    bert_root = tmp_path / "bert"
    bert_root.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_local_bert_root(bert_root)
    (bert_root / "config.json").write_text("{}", encoding="utf-8")
    (bert_root / "model.safetensors").write_bytes(b"weights")
    (bert_root / "vocab.txt").write_text("[UNK]\n", encoding="utf-8")
    _validate_local_bert_root(bert_root)


def test_lae_worker_applies_nms_before_final_top_k(tmp_path: Path) -> None:
    from PIL import Image
    from scripts.integrations.lae_dino_worker import _predict

    image_path = tmp_path / "image.png"
    Image.new("RGB", (40, 40), color="white").save(image_path)

    def inference_detector(_model, _image, **_kwargs):
        return [
            [
                [0.0, 0.0, 20.0, 20.0, 0.9],
                [1.0, 1.0, 19.0, 19.0, 0.8],
                [25.0, 25.0, 39.0, 39.0, 0.7],
            ]
        ]

    args = SimpleNamespace(
        score_threshold=0.3,
        top_k=2,
        nms_threshold=0.5,
        checkpoint_training_regime="test",
        source_revision="test",
        inference_query_mode="target_conditioned_text_prompt",
    )
    response = _predict(
        object(),
        {"id": "one", "image": str(image_path), "target_phrase": "airplanes"},
        args,
        inference_detector,
    )
    assert response["bbox_scores"] == [0.9, 0.7]


def test_lae_worker_proposal_path_is_compatible_with_python39_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    def legacy_zip(*iterables):
        """Model Python 3.9's zip, which rejects the strict keyword."""

        return builtins.zip(*iterables)  # noqa: B905 - deliberately models Python 3.9

    monkeypatch.setattr(lae_dino_worker, "zip", legacy_zip, raising=False)
    monkeypatch.setattr(detector_protocol, "zip", legacy_zip, raising=False)
    response = lae_dino_worker._predict(
        object(),
        {"id": "py39", "image": str(image_path), "target_phrase": "airplanes"},
        SimpleNamespace(score_threshold=0.3, top_k=10, nms_threshold=None),
        lambda *_args, **_kwargs: [[[0.0, 0.0, 5.0, 5.0, 0.9]]],
    )
    assert response["bbox_scores"] == [0.9]


def test_lae_config_patch_sets_absolute_local_bert_path_without_disk_write(tmp_path: Path) -> None:
    bert_root = tmp_path / "bert"
    bert_root.mkdir()
    config = {
        "load_from": "https://example.invalid/bootstrap.pth",
        "model": {
            "language_model": {"name": "../weights/bert-base-uncased"},
            "backbone": {"init_cfg": {"type": "Pretrained", "checkpoint": "url"}},
        },
    }
    patched = _patch_lae_config_for_local_runtime(config, bert_root)
    assert patched["model"]["language_model"]["name"] == str(bert_root)
    assert patched["load_from"] is None
    assert patched["model"]["backbone"]["init_cfg"] is None


def test_lae_worker_skips_validation_dataset_palette_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    config_path = tmp_path / "config.py"
    config_path.write_text("# test config\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    bert_root = tmp_path / "bert"
    bert_root.mkdir()
    (bert_root / "config.json").write_text("{}", encoding="utf-8")
    (bert_root / "model.safetensors").write_bytes(b"weights")
    (bert_root / "vocab.txt").write_text("[UNK]\n", encoding="utf-8")

    config = {
        "load_from": None,
        "model": {
            "language_model": {"name": "legacy"},
            "backbone": {"init_cfg": None},
        },
    }
    captured: dict[str, object] = {}

    class FakeConfig:
        @staticmethod
        def fromfile(_path: str) -> dict[str, object]:
            return config

    def fake_init_detector(cfg, checkpoint_path, **kwargs):
        captured.update({"config": cfg, "checkpoint": checkpoint_path, **kwargs})
        return "model"

    mmdet = ModuleType("mmdet")
    mmdet.__path__ = []  # type: ignore[attr-defined]
    mmdet_apis = ModuleType("mmdet.apis")
    mmdet_apis.init_detector = fake_init_detector  # type: ignore[attr-defined]
    mmengine = ModuleType("mmengine")
    mmengine.__path__ = []  # type: ignore[attr-defined]
    mmengine_config = ModuleType("mmengine.config")
    mmengine_config.Config = FakeConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mmdet", mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", mmdet_apis)
    monkeypatch.setitem(sys.modules, "mmengine", mmengine)
    monkeypatch.setitem(sys.modules, "mmengine.config", mmengine_config)

    model = _load_detector(
        SimpleNamespace(
            source_root=str(source_root),
            config=str(config_path),
            checkpoint=str(checkpoint),
            bert_root=str(bert_root),
            device="cuda:0",
        )
    )
    assert model == "model"
    assert captured["palette"] == "random"
    assert captured["device"] == "cuda:0"
    assert config["model"]["language_model"]["name"] == str(bert_root.resolve())


def test_lae_predict_passes_target_prompt_and_custom_entities(tmp_path: Path) -> None:
    from PIL import Image
    from scripts.integrations.lae_dino_worker import _predict

    image_path = tmp_path / "image.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)
    calls: list[dict[str, object]] = []

    def inference_detector(model, image, **kwargs):
        calls.append({"model": model, "image": image, **kwargs})
        return [[[0.0, 0.0, 5.0, 5.0, 0.9]]]

    args = SimpleNamespace(score_threshold=0.3, top_k=10, nms_threshold=None)
    response = _predict(
        object(),
        {"id": "one", "image": str(image_path), "target_phrase": "Airplanes"},
        args,
        inference_detector,
    )
    assert calls[0]["text_prompt"] == "airplanes"
    assert calls[0]["custom_entities"] is True
    assert response["metadata"]["target_phrase_used_for_detector"] is True
    assert response["metadata"]["inference_query_mode"] == "target_conditioned_text_prompt"
