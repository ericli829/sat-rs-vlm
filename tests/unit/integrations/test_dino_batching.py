from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image
from scripts.integrations import lae_dino_worker as worker

from sat_rs_vlm.integrations.counting import detector_bridge
from sat_rs_vlm.integrations.detectors.lae_dino_sidecar import (
    LAEDinoSidecarProvider,
    SidecarProtocolError,
    _LAESidecarClient,
)
from sat_rs_vlm.integrations.detectors.protocol import ProposalError, ProposalResult
from sat_rs_vlm.integrations.detectors.tile_transport import cpu_prepared_batches
from sat_rs_vlm.integrations.detectors.tiled import TiledProposalProvider


def args(**overrides):
    return argparse.Namespace(
        **{
            "score_threshold": 0.05,
            "top_k": 300,
            "nms_threshold": None,
            "max_batch_size": 16,
            "disable_batch_oom_retry": False,
            **overrides,
        }
    )


def request(tmp_path, count=28):
    path = tmp_path / "tile.bmp"
    Image.new("RGB", (100, 80), "red").save(path)
    return {
        "id": "batch",
        "items": [
            {"id": str(i), "image": str(path), "target_phrase": " CAR "}
            for i in range(count)
        ],
    }


def predictions(items):
    # MMDet 2.x class-box representation; each tile has 20 distinct candidates.
    return [[[[i * 4, 0, i * 4 + 2, 3, 0.9] for i in range(20)]] for _ in items]


@pytest.mark.parametrize("size", [1, 8, 16, 32])
def test_batch_forward_calls_test_step_once_and_preserves_pipeline(monkeypatch, size):
    composed = []
    received = []
    pipeline_cfg = [{"type": "LoadImageFromFile"}, {"type": "original-resize-and-pack"}]

    class Compose:
        def __init__(self, cfg):
            composed.append(cfg)

        def __call__(self, data):
            received.append(data.copy())
            return {"inputs": data["img_path"], "data_samples": data.copy()}

    modules = {
        "torch": {"no_grad": nullcontext},
        "mmcv": {},
        "mmcv.transforms": {"Compose": Compose},
        "mmdet": {},
        "mmdet.utils": {"get_test_pipeline_cfg": lambda _cfg: pipeline_cfg},
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    forwards = []

    def test_step(data):
        forwards.append(data)
        return list(data["data_samples"])

    model = SimpleNamespace(cfg={"test_pipeline": pipeline_cfg}, test_step=test_step)
    infer = worker._build_batch_inference(model)
    items = [
        {"image": f"tile_{i}.bmp", "target_phrase": " White Building "} for i in range(size)
    ]
    result = infer(items)
    assert len(forwards) == 1
    assert len(forwards[0]["inputs"]) == size
    assert composed == [pipeline_cfg]
    assert result == received
    assert [d["img_id"] for d in received] == list(range(size))
    assert all(d["text"] == "white building" and d["custom_entities"] for d in received)
    assert all(Path(d["img_path"]).is_absolute() for d in received)


def test_worker_batches_keep_all_tiles_candidates_and_request_policy(tmp_path):
    req = request(tmp_path)
    req["items"][1]["options"] = {"score_threshold": 0.95}
    calls = []

    def infer(items):
        calls.append(len(items))
        return predictions(items)

    original_args = args()
    result = worker._predict_batch(SimpleNamespace(test_cfg={}), req, original_args, infer)
    assert calls == [16, 12]
    assert [r["id"] for r in result["results"]] == [str(i) for i in range(28)]
    assert result["results"][1]["bbox_list"] == []
    assert all(len(r["bbox_list"]) == 20 for r in result["results"] if r["id"] != "1")
    assert all(r["metadata"]["candidate_pool_complete"] for r in result["results"])
    assert original_args.score_threshold == 0.05
    assert result["results"][0]["metadata"]["effective_batch_size"] == 16
    assert result["results"][-1]["metadata"]["effective_batch_size"] == 12


def test_cuda_oom_splits_retries_same_tiles_and_remembers_limit(tmp_path, monkeypatch):
    req = request(tmp_path)
    calls, releases = [], []

    def infer(items):
        calls.append([item["id"] for item in items])
        if len(items) > 4:
            raise RuntimeError("CUDA out of memory. simulated fixture")
        return predictions(items)

    monkeypatch.setattr(worker, "_release_request_resources", lambda: releases.append(True))
    options = args()
    model = SimpleNamespace(test_cfg={})
    result = worker._predict_batch(model, req, options, infer)
    assert [len(c) for c in calls[:3]] == [16, 8, 4]
    assert [item for chunk in calls[2:] for item in chunk] == [str(i) for i in range(28)]
    assert len(result["results"]) == 28
    assert all(len(r["bbox_list"]) == 20 for r in result["results"])
    assert len(releases) == 2 and options._safe_batch_size == 4
    assert all(r["metadata"]["batch_oom_retries"] == 2 for r in result["results"])
    assert all(r["metadata"]["effective_batch_size"] == 4 for r in result["results"])
    calls.clear()
    worker._predict_batch(model, req, options, infer)
    assert [len(c) for c in calls] == [4] * 7


@pytest.mark.parametrize(
    ("error", "size", "disabled"),
    [("bad model output", 8, False), ("CUDA out of memory", 1, False),
     ("CUDA out of memory", 8, True)],
)
def test_batch_failures_are_errors_not_empty_detections(tmp_path, error, size, disabled):
    calls = []

    def infer(items):
        calls.append(len(items))
        raise RuntimeError(error)

    with pytest.raises(RuntimeError, match=error):
        worker._predict_batch(
            SimpleNamespace(test_cfg={}), request(tmp_path, size),
            args(disable_batch_oom_retry=disabled), infer,
        )
    assert calls == [size]


def test_chunked_text_config_explicitly_falls_back_to_singletons(tmp_path):
    calls = []

    def infer(items):
        calls.append(len(items))
        return predictions(items)

    result = worker._predict_batch(
        SimpleNamespace(test_cfg={"chunked_size": 256}), request(tmp_path, 3), args(), infer,
    )
    assert calls == [1, 1, 1]
    assert all(r["metadata"]["chunked_text_serial"] for r in result["results"])


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "short_output"])
def test_worker_batch_protocol_rejects_invalid_items(tmp_path, invalid):
    req = request(tmp_path, 2)
    if invalid == "empty":
        req["items"] = []
    elif invalid == "duplicate":
        req["items"][1]["id"] = req["items"][0]["id"]
    with pytest.raises((ValueError, ProposalError)):
        worker._predict_batch(SimpleNamespace(test_cfg={}), req, args(), lambda _items: [])


def test_batch_rescales_and_clamps_each_image_separately(tmp_path):
    req = request(tmp_path, 2)
    small = tmp_path / "small.bmp"
    Image.new("RGB", (30, 20), "blue").save(small)
    req["items"][1]["image"] = str(small)
    # The model returns per-image rescaled boxes; validation must use that image.
    result = worker._predict_batch(
        SimpleNamespace(test_cfg={}), req, args(),
        lambda items: [[[[0, 0, 200, 200, 0.9]]] for _ in items],
    )
    assert [r["bbox_list"] for r in result["results"]] == [
        [[0.0, 0.0, 100.0, 80.0]], [[0.0, 0.0, 30.0, 20.0]],
    ]


def test_request_options_do_not_change_shared_defaults():
    defaults = args()
    policy = worker._request_options(defaults, {"options": {
        "score_threshold": 0.25, "top_k": 100, "nms_threshold": 0.4,
    }})
    assert (defaults.score_threshold, defaults.top_k, defaults.nms_threshold) == (0.05, 300, None)
    assert (policy.score_threshold, policy.top_k, policy.nms_threshold) == (0.25, 100, 0.4)
    for bad in ({"score_threshold": float("nan")}, {"top_k": 0}, {"top_k": True},
                {"nms_threshold": 2}, {"unknown_option": 1}):
        with pytest.raises(ValueError):
            worker._request_options(defaults, {"options": bad})


def sidecar_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bert = tmp_path / "bert"
    bert.mkdir()
    config = tmp_path / "config.py"
    config.write_text("# fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"test fixture")
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, os, sys\n"
        "for line in sys.stdin:\n"
        "    q = json.loads(line)\n"
        "    r = {'id': q['id'], 'status': 'ok'}\n"
        "    if q.get('operation') == 'initialize':\n"
        "        r['model_inventory'] = {'parameter_count': 123, 'worker_pid': os.getpid()}\n"
        "    else:\n"
        "        def result(item):\n"
        "            return {'id': item['id'], 'status': 'ok', 'bbox_list': [],\n"
        "                    'bbox_scores': [], 'metadata': {'image_width': 100,\n"
        "                    'image_height': 80, 'worker_pid': os.getpid(),\n"
        "                    'options': item.get('options'), 'candidate_truncated': False}}\n"
        "        if q.get('operation') == 'predict_batch':\n"
        "            r['results'] = [result(item) for item in q['items']]\n"
        "        else:\n"
        "            r = result(q)\n"
        "    print(json.dumps(r), flush=True)\n",
        encoding="utf-8",
    )
    return dict(source_root=str(source), config_path=str(config), checkpoint=str(checkpoint),
                bert_root=str(bert), worker_python=sys.executable, worker_script=str(script),
                device="cpu", batch_size=16, parallel_workers=5)


def test_locate_count_share_one_worker_with_independent_policy_and_lifetime(tmp_path):
    config = sidecar_config(tmp_path)
    locate = LAEDinoSidecarProvider(
        {**config, "score_threshold": 0.05, "top_k": 100}, provider_name="lae_dino_lae1m",
    )
    count = LAEDinoSidecarProvider(
        {**config, "score_threshold": 0.25, "top_k": 300, "nms_threshold": 0.4},
        provider_name="lae_dino_lae1m",
    )
    try:
        locate.preload()
        count.preload()
        assert locate._client is count._client
        process = locate._client.process
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda p: p.predict(tmp_path / "tile.bmp", "car"), [locate, count] * 2,
            ))
        assert {r.metadata["worker_pid"] for r in results} == {process.pid}
        assert [r.metadata["options"]["top_k"] for r in results] == [100, 300] * 2
        assert [r.metadata["options"]["score_threshold"] for r in results] == [0.05, 0.25] * 2
        assert all(r.metadata["parallel_workers"] == 1 for r in results)
        assert locate.telemetry_parameter_count == count.telemetry_parameter_count == 123
        locate.close()
        assert process.poll() is None
        assert len(count.predict_batch([tmp_path / "tile.bmp"] * 33, "car")) == 33
        with pytest.raises(ProposalError, match="closed"):
            locate.predict(tmp_path / "tile.bmp", "car")
    finally:
        locate.close()
        count.close()
    assert process.poll() is not None


@pytest.mark.parametrize("fault", ["none", "duplicate", "unknown", "missing", "failed"])
def test_batch_client_validates_ids_and_restores_order(tmp_path, monkeypatch, fault):
    client = _LAESidecarClient(sidecar_config(tmp_path))

    def exchange(req):
        results = [dict(id=i["id"], status="ok") for i in reversed(req["items"])]
        if fault == "duplicate":
            results[1]["id"] = results[0]["id"]
        elif fault == "unknown":
            results[0]["id"] = "unknown"
        elif fault == "missing":
            results.pop()
        elif fault == "failed":
            results[0]["status"] = "failed"
        for result in results:
            result["original_index"] = next(
                (i for i, item in enumerate(req["items"]) if item["id"] == result["id"]), -1,
            )
        return {"id": req["id"], "status": "ok", "results": results}

    monkeypatch.setattr(client, "_exchange", exchange)
    if fault == "none":
        result = client.request_batch([tmp_path / "tile.bmp"] * 3, "CAR", options={})
        assert [r["original_index"] for r in result] == [0, 1, 2]
    else:
        with pytest.raises(SidecarProtocolError):
            client.request_batch([tmp_path / "tile.bmp"] * 3, "car", options={})


def test_tiled_batch_coordinates_order_complete_candidates_and_cleanup(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (400, 10), "red").save(path)
    calls, paths, threads = [], [], []

    class Provider:
        inference_batch_size = 16

        def predict(self, *_args):
            pytest.fail("single tile inference should not run")

        def predict_batch(self, batch, phrase):
            assert phrase == "building"
            assert all(p.exists() for p in batch)
            calls.append(len(batch))
            paths.extend(batch)
            threads.append(threading.get_ident())
            return [ProposalResult([[1, 2, 5, 6]], [0.9], 1, "test", "test",
                                   {"candidate_pool_complete": True}) for _ in batch]

        def close(self):
            pass

    provider = TiledProposalProvider(
        Provider(), {"tile_size": 10, "overlap_ratio": 0, "parallel_workers": 5},
        base_provider_name="test",
    )
    result = provider.predict(path, "building")
    assert calls == [16, 16, 8] and len(set(threads)) == 1
    assert result.boxes_xyxy == [[i * 10 + 1, 2, i * 10 + 5, 6] for i in range(40)]
    assert result.metadata["candidate_pool_complete"] is True
    assert result.metadata["parallel_workers"] == 1 and result.metadata["batch_count"] == 3
    assert [r["tile_id"] for r in result.metadata["tiles"]] == list(range(40))
    assert all(not p.exists() for p in paths)


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("exit_kind", ["all", "break", "inference_error", "prepare_error"])
def test_cpu_batches_bound_resources_release_partial_batches_and_close_source(prefetch, exit_kind):
    prepared, released, closed = [], [], []
    next_ready = threading.Event()

    def source():
        try:
            yield from range(100)
        finally:
            closed.append(True)

    def prepare(i):
        if i == 5 and exit_kind == "prepare_error":
            raise ValueError("prepare failed")
        prepared.append(i)
        if i == 7:
            next_ready.set()
        return i

    expected = ValueError if exit_kind == "prepare_error" else RuntimeError
    context = pytest.raises(expected) if exit_kind.endswith("error") else nullcontext()
    with context:
        with cpu_prepared_batches(source(), prepare, released.append,
                                  batch_size=4, prefetch=prefetch) as batches:
            for index, batch in enumerate(batches):
                assert len(prepared) - len(released) <= (8 if prefetch else 4)
                if prefetch and index == 0 and exit_kind != "prepare_error":
                    assert next_ready.wait(3)
                    assert prepared == list(range(8))
                if exit_kind == "break":
                    break
                if exit_kind == "inference_error":
                    raise RuntimeError("inference failed")
                assert batch == list(range(index * 4, index * 4 + 4))
    assert sorted(prepared) == sorted(released)
    assert len(released) == len(set(released)) and closed == [True]


@pytest.mark.parametrize("exit_kind", ["all", "break", "error"])
def test_count_batch_handles_image_ownership_query_groups_coordinates_and_cleanup(exit_kind):
    calls, images, paths = [], [], []

    def requests():
        for i in range(21):
            image = Image.new("RGB", (10, 10), "blue")
            images.append(image)
            try:
                yield detector_bridge.DetectionRequest(
                    image=image, texts="car" if i < 17 else "ship",
                    target=SimpleNamespace(name="object"),
                    tile=SimpleNamespace(crop_xyxy=(i * 20, 0, i * 20 + 20, 20),
                                         tile_id=str(i), scale_id="native"),
                )
            finally:
                image.close()

    class Provider:
        inference_batch_size = 16

        def predict_batch(self, batch, phrase):
            calls.append((len(batch), phrase))
            paths.extend(batch)
            for p in batch:
                with Image.open(p) as image:
                    assert image.getpixel((0, 0)) == (0, 0, 255)
            if exit_kind == "error":
                raise ProposalError("simulated batch failure")
            return [ProposalResult([[1, 2, 5, 6]], [0.8], 1, "test", "test") for _ in batch]

        def close(self):
            pass

    bridge = detector_bridge.CountingProposalDetectorBridge(Provider())
    stream = bridge.detect_many(requests())
    try:
        if exit_kind == "error":
            with pytest.raises(RuntimeError, match="batch failure"):
                list(stream)
        elif exit_kind == "break":
            next(stream)
        else:
            responses = list(stream)
            assert calls == [(16, "car"), (1, "car"), (4, "ship")]
            assert bridge.call_count == 21
            assert [r.detections[0].bbox_xyxy_global for r in responses] == [
                (i * 20 + 2, 4, i * 20 + 10, 12) for i in range(21)
            ]
    finally:
        stream.close()
        bridge.close()
    assert all(not p.exists() for p in paths)
    for image in images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


def test_worker_main_initialize_and_batch_protocol_dispatch(tmp_path, monkeypatch, capsys):
    req = request(tmp_path, 3)
    model = SimpleNamespace(test_cfg={}, parameters=lambda: [SimpleNamespace(numel=lambda: 7)])
    apis = ModuleType("mmdet.apis")
    apis.inference_detector = lambda *_args, **_kwargs: pytest.fail("not a batch")
    monkeypatch.setitem(sys.modules, "mmdet", ModuleType("mmdet"))
    monkeypatch.setitem(sys.modules, "mmdet.apis", apis)
    monkeypatch.setattr(worker, "parse_args", args)
    monkeypatch.setattr(worker, "_load_detector", lambda _args: model)
    monkeypatch.setattr(worker, "_build_batch_inference", lambda _model: predictions)
    monkeypatch.setattr(worker, "_release_request_resources", lambda: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, [
        {"id": "init", "operation": "initialize"}, {**req, "operation": "predict_batch"},
    ]))))
    assert worker.main() == 0
    init, batch = map(json.loads, capsys.readouterr().out.splitlines())
    assert init["status"] == "ok" and init["model_inventory"]["parameter_count"] == 7
    assert init["model_inventory"]["worker_pid"] == os.getpid()
    assert len(batch["results"]) == 3 and batch["id"] == "batch"
    ast.parse(Path(worker.__file__).read_text(encoding="utf-8"), feature_version=(3, 9))
