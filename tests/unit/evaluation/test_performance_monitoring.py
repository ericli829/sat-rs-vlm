from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sat_rs_vlm.evaluation.inference import timed_prediction_with_telemetry
from sat_rs_vlm.evaluation.performance import (
    PerformanceMonitor,
    environment_metadata,
    model_resource_metadata,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
        self.shape = shape
        self.device = device

    def to(self, device: object) -> FakeTensor:
        self.device = str(device)
        return self


class FakeOutputIds:
    def __getitem__(self, key: object) -> list[list[int]]:
        assert key == (slice(None), slice(3, None))
        return [[9, 10]]


def test_timed_prediction_collects_token_count_and_end_to_end_latency() -> None:
    input_ids = FakeTensor((1, 3))

    class Model:
        def get_input_embeddings(self) -> Any:
            return SimpleNamespace(weight=FakeTensor((1,), "cpu"))

        def generate(self, **kwargs: Any) -> FakeOutputIds:
            criteria = kwargs.get("stopping_criteria")
            if criteria is not None:
                criteria([[1, 2, 3, 9]], None)
            return FakeOutputIds()

    class Processor:
        def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
            assert token_ids == [[9, 10]]
            return ["timed answer"]

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
        is_tensor=lambda value: isinstance(value, FakeTensor),
        inference_mode=lambda: nullcontext(),
    )
    prediction, timing = timed_prediction_with_telemetry(
        Model(),
        Processor(),
        lambda batch: {"input_ids": input_ids},  # type: ignore[arg-type]
        {"id": "sample", "task_type": "vqa"},
        {"do_sample": False},
        torch,
    )

    assert prediction == "timed answer"
    assert timing.end_to_end_latency_ms >= timing.generation_latency_ms
    assert timing.output_token_count == 2
    assert timing.generation_tokens_per_second is not None


def test_performance_monitor_reports_statistics_and_environment() -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monitor = PerformanceMonitor(torch)
    monitor.start()
    monitor.record(
        "vqa",
        {
            "ttft_ms": 5.0,
            "generation_tokens_per_second": 20.0,
            "decode_tokens_per_second": 30.0,
            "output_token_count": 4,
        },
        system_latency_ms=10.0,
    )
    monitor.record(
        "vqa",
        {
            "ttft_ms": 7.0,
            "generation_tokens_per_second": 10.0,
            "decode_tokens_per_second": None,
            "output_token_count": 1,
        },
        system_latency_ms=20.0,
    )
    report = monitor.finish(
        requested_samples=2,
        completed_samples=2,
        failed_samples=0,
        warmup_samples=1,
        startup_and_model_load_ms=100.0,
        model_load_ms=80.0,
        config={"generation": {"max_new_tokens": 8}},
        environment=environment_metadata(torch, model_config={"torch_dtype": "bfloat16"}),
        model_resources={"loaded_model_logical_parameter_count": None},
        batch_size=1,
        repeats=1,
    )

    assert report["latency_ms"]["mean"] == 15.0
    assert report["latency_ms"]["p95"] == 20.0
    assert report["ttft_ms"]["samples"] == 2
    assert report["decode_tokens_per_second"]["samples"] == 1
    assert report["by_task"]["vqa"]["output_token_count"]["samples"] == 2
    assert report["environment"]["accelerator"]["cuda_available"] is False
    assert report["run"]["batch_size"] == 1
    assert report["run"]["repeats"] == 1


def test_model_resource_metadata_counts_local_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"1234")

    class Parameter:
        def numel(self) -> int:
            return 7

    class Model:
        def parameters(self) -> list[Parameter]:
            return [Parameter(), Parameter()]

    resources = model_resource_metadata(Model(), model_config={"base_model": str(model_dir)})

    assert resources["loaded_model_logical_parameter_count"] == 14
    assert resources["local_model_storage_bytes"] == 4
