import importlib.util
import time

import pytest

from sat_rs_vlm.infrastructure.profiler import InferenceProfiler


def test_profiler_records_latency() -> None:
    with InferenceProfiler(backend="mock", device="cpu") as profiler:
        time.sleep(0.001)
    payload = profiler.to_dict()
    assert payload["latency_ms"] is not None
    assert payload["latency_ms"] >= 0
    assert payload["backend"] == "mock"
    assert payload["device"] == "cpu"


def test_profiler_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object:
        if name == "torch":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(
        "sat_rs_vlm.infrastructure.profiler.importlib.util.find_spec",
        fake_find_spec,
    )
    with InferenceProfiler(backend="huggingface", device="cpu") as profiler:
        pass
    payload = profiler.to_dict()
    assert payload["cuda_available"] is False
    assert payload["latency_ms"] is not None


def test_profiler_degrades_when_torch_dll_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sat_rs_vlm.infrastructure.profiler.importlib.util.find_spec",
        lambda name: object(),
    )

    def fail_import(name: str) -> object:
        raise OSError(f"cannot load {name} c10.dll")

    monkeypatch.setattr(
        "sat_rs_vlm.infrastructure.profiler.importlib.import_module",
        fail_import,
    )

    with InferenceProfiler(backend="huggingface", device="cuda") as profiler:
        pass

    payload = profiler.to_dict()
    assert payload["cuda_available"] is False
    assert "c10.dll" in payload["torch_error"]
