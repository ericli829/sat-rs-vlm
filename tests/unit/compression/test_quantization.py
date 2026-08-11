from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.quantization.artifacts import to_json_safe, write_json_report
from sat_rs_vlm.quantization.benchmark import (
    _iter_task_batches,
    assert_comparable_sample_ids,
    planned_variants,
    run_benchmark,
    select_evaluation_samples,
    validate_assets,
)
from sat_rs_vlm.quantization.config import (
    QuantizationExperimentConfig,
    load_quantization_config,
)
from sat_rs_vlm.quantization.quantizer import (
    QuantizationBackend,
    UnsupportedQuantizationError,
    create_backend,
    quantize_dynamic_linear,
    register_backend,
    verify_selective_bnb_int8_modules,
)
from sat_rs_vlm.quantization.report import comparison_summary
from sat_rs_vlm.utils.jsonl import write_jsonl


def _config(tmp_path: Path, *, backend: str = "torch_dynamic_int8") -> QuantizationExperimentConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model"
    images = tmp_path / "images"
    model.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    for name in ("a.png", "b.png"):
        (images / name).write_bytes(b"image")
    eval_file = tmp_path / "eval.jsonl"
    write_jsonl(
        eval_file,
        [
            {
                "id": "multi",
                "task_type": "change_detection",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "a.png"},
                            {"type": "image", "image": "b.png"},
                            {"type": "text", "text": "What changed?"},
                        ],
                    },
                    {"role": "assistant", "content": "A building appeared."},
                ],
            }
        ],
    )
    return QuantizationExperimentConfig.model_validate(
        {
            "model": {"base_model": str(model), "processor_id": str(model)},
            "quantization": {
                "backend": backend,
                "device": "cpu" if backend != "bnb_int8" else "cuda",
            },
            "data": {"eval_file": str(eval_file), "image_root": str(images)},
            "output": {"output_dir": str(tmp_path / "output")},
        }
    )


def test_backend_config_and_unknown_backend(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.quantization.backend == "torch_dynamic_int8"
    with pytest.raises(ValueError, match="Unknown quantization backend"):
        create_backend("wrong")
    with pytest.raises(ValueError, match="requires device='cpu'"):
        QuantizationExperimentConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "quantization": {"backend": "torch_dynamic_int8", "device": "cuda"},
            }
        )


def test_new_method_alias_and_merged_model_source(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = config.model_dump(mode="python")
    payload["model"]["merged_model"] = "merged"
    payload["quantization"] = {"method": "dynamic_int8", "device": "cpu"}

    normalized = QuantizationExperimentConfig.model_validate(payload)

    assert normalized.quantization.backend == "torch_dynamic_int8"
    assert normalized.quantization.method == "dynamic_int8"
    assert normalized.model.model_source == "merged"

    payload["quantization"] = {
        "method": "dynamic_int8",
        "backend": "bnb_int8",
        "device": "cuda",
    }
    with pytest.raises(ValueError, match="conflicts with backend"):
        QuantizationExperimentConfig.model_validate(payload)


def test_batched_benchmark_requires_explicit_batch_latency_semantics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = config.model_dump(mode="python")
    payload["benchmark"] = {
        "inference_batch_size": 4,
        "latency_scope": "batch_amortized_per_sample",
    }
    resolved = QuantizationExperimentConfig.model_validate(payload)
    assert resolved.benchmark.inference_batch_size == 4

    payload["benchmark"]["latency_scope"] = "single_sample_end_to_end"
    with pytest.raises(ValueError, match="requires inference_batch_size=1"):
        QuantizationExperimentConfig.model_validate(payload)


def test_task_batches_keep_same_task_samples_together() -> None:
    dataset = [
        {"id": "vqa-1", "task_type": "vqa"},
        {"id": "caption-1", "task_type": "captioning"},
        {"id": "vqa-2", "task_type": "vqa"},
    ]

    batches = _iter_task_batches(dataset, batch_size=2)

    assert [(task, [sample["id"] for _, sample in rows]) for task, rows in batches] == [
        ("vqa", ["vqa-1", "vqa-2"]),
        ("captioning", ["caption-1"]),
    ]


def test_legacy_quantization_import_path_remains_available() -> None:
    from sat_rs_vlm.compression.quantization import create_backend as legacy_create_backend

    assert legacy_create_backend("baseline").name == "baseline"


def test_future_backend_registration_rejects_accidental_overwrite() -> None:
    class FutureBackend(QuantizationBackend):
        name = "future_int4_test"
        device = "cpu"

        def load_model(self, config, modules, *, quantized):  # type: ignore[no-untyped-def]
            return object()

        def compression_metadata(  # type: ignore[no-untyped-def]
            self, model, torch, *, quantized
        ):
            return {"backend": self.name}

    register_backend("future_int4_test", FutureBackend, replace=True)
    assert create_backend("FUTURE_INT4_TEST").name == "future_int4_test"
    with pytest.raises(ValueError, match="already registered"):
        register_backend("future_int4_test", FutureBackend)


def test_load_config_expands_existing_environment(tmp_path: Path) -> None:
    config_file = tmp_path / "quant.yaml"
    config_file.write_text(
        """
model: {base_model: "${MODEL}"}
quantization: {backend: torch_dynamic_int8, device: cpu}
data: {eval_file: "${EVAL}", image_root: "${IMAGES}"}
output: {output_dir: "${OUTPUT}"}
""",
        encoding="utf-8",
    )
    config = load_quantization_config(
        config_file,
        environ={
            "MODEL": "model",
            "EVAL": "eval.jsonl",
            "IMAGES": "images",
            "OUTPUT": "output",
        },
    )
    assert config.model.base_model == "model"


def test_messages_manifest_supports_multiple_images_and_missing_image(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset, manifest, _, _ = validate_assets(
        config,
        create_backend("torch_dynamic_int8"),
        project_root=tmp_path,
    )
    assert len(dataset) == 1
    assert len(manifest[0]["images"]) == 2
    assert manifest[0]["question"] == "What changed?"

    Path(manifest[0]["images"][1]).unlink()
    with pytest.raises(FileNotFoundError, match="multi"):
        validate_assets(config, create_backend("torch_dynamic_int8"), project_root=tmp_path)


def test_empty_eval_and_unverified_adapter_fail_explicitly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    Path(config.data.eval_file).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no samples"):
        validate_assets(config, create_backend("torch_dynamic_int8"), project_root=tmp_path)

    config = _config(tmp_path / "second")
    config.model.adapter_path = str(tmp_path / "adapter")
    with pytest.raises(FileNotFoundError, match="adapter"):
        validate_assets(config, create_backend("torch_dynamic_int8"), project_root=tmp_path)


def test_skip_baseline_and_sample_fairness() -> None:
    assert planned_variants("bnb_int8", skip_baseline=True) == ("quantized",)
    with pytest.raises(ValueError, match="leaves no variant"):
        planned_variants("baseline", skip_baseline=True)
    assert comparison_summary(None, {"latency_ms": {"mean": 1.0}})["speedup"] is None
    assert_comparable_sample_ids({"sample_ids": ["a"]}, {"sample_ids": ["a"]})
    with pytest.raises(RuntimeError, match="different sample IDs"):
        assert_comparable_sample_ids({"sample_ids": ["a"]}, {"sample_ids": ["b"]})


def test_quantization_sampling_filters_before_limiting_and_balances_tasks() -> None:
    rows = [
        {"id": "caption-1", "task_type": "captioning"},
        {"id": "caption-2", "task_type": "captioning"},
        {"id": "vqa-1", "task_type": "vqa"},
        {"id": "vqa-2", "task_type": "vqa"},
        {"id": "change-1", "task_type": "change_detection"},
    ]

    selected = select_evaluation_samples(
        rows,
        allowed_tasks={"vqa", "change_detection"},
        max_samples=2,
        strategy="stratified",
        seed=42,
        samples_per_task={"vqa": 1, "change_detection": 1},
    )

    assert {row["task_type"] for row in selected} == {"vqa", "change_detection"}


def test_dynamic_int8_quantizes_toy_linear_model() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    quantized = quantize_dynamic_linear(model, torch)
    assert "quantized.dynamic" in quantized[0].__class__.__module__


def test_selective_bnb_verification_rejects_missed_or_unexpected_modules() -> None:
    Linear8bitLt = type("Linear8bitLt", (), {"__module__": "bitsandbytes.nn.modules"})

    class FloatLinear:
        pass

    class FakeModel:
        def __init__(self, target: object, skipped: object) -> None:
            self.target = target
            self.skipped = skipped

        def named_modules(self) -> list[tuple[str, object]]:
            return [("", self), ("target", self.target), ("skipped", self.skipped)]

    verify_selective_bnb_int8_modules(
        FakeModel(Linear8bitLt(), FloatLinear()),
        target_module_names=("target",),
        skipped_module_names=("skipped",),
    )

    with pytest.raises(RuntimeError, match="not converted"):
        verify_selective_bnb_int8_modules(
            FakeModel(FloatLinear(), FloatLinear()),
            target_module_names=("target",),
            skipped_module_names=("skipped",),
        )

    with pytest.raises(RuntimeError, match="unexpectedly converted"):
        verify_selective_bnb_int8_modules(
            FakeModel(Linear8bitLt(), Linear8bitLt()),
            target_module_names=("target",),
            skipped_module_names=("skipped",),
        )


def test_bnb_metadata_reports_mixed_weights_and_actual_int8_count() -> None:
    Linear8bitLt = type("Linear8bitLt", (), {"__module__": "bitsandbytes.nn.modules"})

    class Parameter:
        dtype = "torch.bfloat16"

    class FakeModel:
        def parameters(self):  # type: ignore[no-untyped-def]
            return iter([Parameter()])

        def named_modules(self) -> list[tuple[str, object]]:
            return [("", self), ("quantized", Linear8bitLt()), ("float", object())]

    class Torch:
        class version:
            cuda = "13.0"

    metadata = create_backend("bnb_int8").compression_metadata(FakeModel(), Torch(), quantized=True)
    assert metadata["weight_dtype"] == "mixed_int8_bfloat16"
    assert metadata["int8_linear_count"] == 1


def test_json_report_is_safe_and_report_file_is_persisted(tmp_path: Path) -> None:
    class Scalar:
        def item(self) -> int:
            return 7

    report_file = tmp_path / "report.json"
    payload = {"path": Path("x"), "scalar": Scalar(), "missing": None}
    write_json_report(report_file, payload)
    written = json.loads(report_file.read_text(encoding="utf-8"))
    assert written["report_file"] == str(report_file)
    assert to_json_safe(payload) == {"path": "x", "scalar": 7, "missing": None}


def test_dry_run_validates_assets_without_loading_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = run_benchmark(
        config,
        create_backend("torch_dynamic_int8"),
        project_root=tmp_path,
        dry_run=True,
    )
    assert report["success"] is True
    assert report["dry_run"] is True
    assert (Path(config.output.output_dir) / "sample_manifest.jsonl").is_file()


def test_dynamic_adapter_combination_is_unsupported(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    config.model.adapter_path = str(adapter)
    with pytest.raises(UnsupportedQuantizationError, match="not verified"):
        create_backend("torch_dynamic_int8").validate(config)


def test_baseline_metadata_reports_actual_device_and_dtype() -> None:
    class Parameter:
        dtype = "torch.float16"
        device = "cuda:1"

    class Model:
        def parameters(self):  # type: ignore[no-untyped-def]
            return iter([Parameter()])

    metadata = create_backend("baseline").compression_metadata(Model(), object(), quantized=False)
    assert metadata["device"] == "cuda:1"
    assert metadata["weight_dtype"] == "float16"
    assert metadata["compute_dtype"] == "float16"
