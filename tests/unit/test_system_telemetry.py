from __future__ import annotations

from types import SimpleNamespace

import pytest

from sat_rs_vlm.infrastructure.telemetry import (
    GenerationTelemetry,
    SystemTelemetry,
    collect_model_inventory,
    collect_provider_inventory,
    collect_runtime_environment,
    visual_input_telemetry,
)


class FakeCuda:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.sync_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Test GPU"

    @staticmethod
    def get_device_properties(index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(total_memory=8 * 1024 * 1024)

    def synchronize(self) -> None:
        self.sync_calls += 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    @staticmethod
    def max_memory_allocated() -> int:
        return 4 * 1024 * 1024

    @staticmethod
    def max_memory_reserved() -> int:
        return 6 * 1024 * 1024


def test_system_telemetry_records_resources_and_measurement_semantics() -> None:
    cuda = FakeCuda()
    torch = SimpleNamespace(cuda=cuda)

    with SystemTelemetry(
        "test_scope",
        torch_module=torch,
        reset_cuda_peaks=True,
        cpu_sample_interval_s=0,
    ) as monitor:
        pass

    payload = monitor.to_dict()
    assert payload["scope"] == "test_scope"
    assert payload["success"] is True
    assert payload["timing_ms"]["e2e"] >= 0
    assert payload["resources"]["peak_cpu_rss_mb"] is not None
    assert payload["resources"]["peak_gpu_allocated_mb"] == 4.0
    assert payload["resources"]["peak_gpu_reserved_mb"] == 6.0
    assert payload["measurement"]["cuda_synchronized"] is True
    assert payload["measurement"]["cuda_peak_stats_reset"] is True
    assert cuda.reset_calls == 1
    assert cuda.sync_calls == 2


def test_system_telemetry_records_failure_without_swallowing_it() -> None:
    monitor = SystemTelemetry("failure", cpu_sample_interval_s=0)
    with pytest.raises(RuntimeError, match="expected"):
        with monitor:
            raise RuntimeError("expected")

    assert monitor.to_dict()["success"] is False
    assert monitor.to_dict()["error_type"] == "RuntimeError"


def test_collect_model_inventory_counts_parameters_and_unique_storage(tmp_path) -> None:
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"123456")

    class Parameter:
        dtype = "torch.float16"

        @staticmethod
        def numel() -> int:
            return 3

        @staticmethod
        def element_size() -> int:
            return 2

    model = SimpleNamespace(parameters=lambda: iter((Parameter(), Parameter())))
    inventory = collect_model_inventory(model, [tmp_path, weights])

    assert inventory["parameter_count"] == 6
    assert inventory["loaded_parameter_bytes"] == 12
    assert inventory["parameters_by_dtype"] == {"float16": 6}
    assert inventory["local_model_storage_bytes"] == 6


def test_collect_runtime_environment_reports_cuda_device(monkeypatch) -> None:
    cuda = FakeCuda()
    torch = SimpleNamespace(
        cuda=cuda,
        __version__="2.test",
        version=SimpleNamespace(cuda="13.test"),
    )
    monkeypatch.setattr(
        "sat_rs_vlm.infrastructure.telemetry._nvidia_driver_version",
        lambda: "999.test",
    )

    environment = collect_runtime_environment(torch)

    assert environment["gpu"]["count"] == 1
    assert environment["gpu"]["devices"][0]["name"] == "Test GPU"
    assert environment["gpu"]["driver_version"] == "999.test"
    assert environment["gpu"]["cuda_runtime"] == "13.test"
    assert environment["software"]["torch"] == "2.test"


def test_collect_provider_inventory_keeps_lazy_and_non_model_states_explicit(
    tmp_path,
) -> None:
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"weights")

    class Parameter:
        dtype = "torch.float32"

        @staticmethod
        def numel() -> int:
            return 2

        @staticmethod
        def element_size() -> int:
            return 4

    loaded_model = SimpleNamespace(parameters=lambda: iter((Parameter(),)))
    loaded = SimpleNamespace(
        provider_name="loaded_vlm",
        role="answer",
        model_config=SimpleNamespace(model_id=str(weights)),
        _engine=SimpleNamespace(_model=loaded_model),
    )
    lazy = SimpleNamespace(
        provider_name="lazy_vlm",
        role="route",
        model_config=SimpleNamespace(model_id="remote/model"),
        _engine=None,
    )
    detector = SimpleNamespace(provider_name="detector")

    inventory = collect_provider_inventory([loaded, lazy, detector])

    assert inventory["total_parameter_count"] is None
    assert inventory["known_parameter_count"] == 2
    assert inventory["parameter_accounting_status"] == "partial"
    assert [item["status"] for item in inventory["models"]] == [
        "loaded",
        "declared_only",
        "not_a_model",
    ]


def test_generation_telemetry_reports_ttft_decode_rate_and_visual_grid() -> None:
    telemetry = GenerationTelemetry()
    telemetry.start()
    telemetry.start_preprocess()
    telemetry.finish_preprocess()
    telemetry.start_generation()
    telemetry.mark_first_token()
    telemetry.finish_generation(6)
    telemetry.start_decode()
    telemetry.finish_decode()
    telemetry.output_token_counts = [5]
    telemetry.vision_input = visual_input_telemetry(
        [[SimpleNamespace(size=(100, 50))]], [[1, 4, 6]]
    )

    payload = telemetry.to_dict()

    assert payload["timing_ms"]["ttft"] is not None
    assert payload["tokens"]["generated"] == 6
    assert payload["tokens"]["decode_tokens_per_second"] is not None
    assert payload["vision_input"]["processed_size"] == [[[96, 64]]]
    assert payload["vision_input"]["visual_token_count"] == 6
