from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.compression.quantization.backends import (
    UnsupportedQuantizationError,
    create_backend,
    quantize_dynamic_linear,
)
from sat_rs_vlm.compression.quantization.benchmark import (
    assert_comparable_sample_ids,
    planned_variants,
    run_benchmark,
    validate_assets,
)
from sat_rs_vlm.compression.quantization.config import (
    QuantizationExperimentConfig,
    load_quantization_config,
)
from sat_rs_vlm.compression.quantization.manifest import to_json_safe, write_json_report
from sat_rs_vlm.compression.quantization.report import comparison_summary
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


def test_dynamic_int8_quantizes_toy_linear_model() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    quantized = quantize_dynamic_linear(model, torch)
    assert "quantized.dynamic" in quantized[0].__class__.__module__


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
