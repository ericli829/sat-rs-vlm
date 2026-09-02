from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sat_rs_vlm.evaluation.performance import (
    PerformanceMonitor,
    environment_metadata,
    model_resource_metadata,
)


def test_performance_monitor_reports_latency_memory_and_input_statistics() -> None:
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
        input_profile={
            "image_count": 1,
            "visual_token_count": 256,
            "visual_token_count_status": "measured",
        },
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
    assert report["input_profile"]["visual_token_count"]["mean"] == 256.0
    assert report["environment"]["accelerator"]["cuda_available"] is False
    assert report["run"]["batch_size"] == 1


def test_model_resource_metadata_counts_local_model_directory(tmp_path: Path) -> None:
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
