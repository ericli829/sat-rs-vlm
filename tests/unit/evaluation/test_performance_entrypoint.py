from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sat_rs_vlm.evaluation.inference import PredictionTiming


def _load_evaluate_script() -> Any:
    """当前工作树缺少真实loader时，为入口单测提供最小导入替身。"""

    loader_name = "sat_rs_vlm.models.qwen3vl_loader"
    if loader_name not in sys.modules:
        loader = types.ModuleType(loader_name)
        loader.load_qwen3vl = lambda **kwargs: (None, None)
        loader.validate_local_adapter = lambda *args, **kwargs: None
        loader.compatible_model_class = lambda transformers: None
        sys.modules[loader_name] = loader
    sys.modules.pop("scripts.evaluate_rs_vlm", None)
    from scripts import evaluate_rs_vlm

    return evaluate_rs_vlm


def test_evaluate_writes_performance_report(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_evaluate_script()
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("{}\n", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()
    output_dir = tmp_path / "output"
    sample = {
        "id": "one",
        "task_type": "vqa",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        "metadata": {},
    }
    config = {
        "model": {"torch_dtype": "bfloat16", "device_map": "cpu"},
        "data": {
            "eval_file": str(eval_file),
            "image_root": str(image_root),
            "max_seq_length": 64,
            "max_eval_samples": 1,
        },
        "generation": {"do_sample": False, "max_new_tokens": 8},
        "performance": {"enabled": True, "warmup_samples": 1, "continue_on_error": True},
        "output": {},
    }
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    timing = PredictionTiming(
        end_to_end_latency_ms=12.0,
        generation_latency_ms=10.0,
        ttft_ms=4.0,
        decode_latency_ms=1.0,
        output_token_count=3,
        generation_tokens_per_second=300.0,
        decode_tokens_per_second=200.0,
        input_profile={
            "image_count": 0,
            "visual_token_count": None,
            "visual_token_count_status": "unresolved_no_image_grid",
        },
    )
    monkeypatch.setattr(module, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        module,
        "safe_import_model_dependencies",
        lambda **kwargs: {"torch": fake_torch},
    )
    monkeypatch.setattr(module, "load_model", lambda config, modules: (object(), object()))
    monkeypatch.setattr(module, "Qwen3VLDataset", lambda path, limit: [sample])
    monkeypatch.setattr(module, "Qwen3VLDataCollator", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "timed_prediction", lambda *args, **kwargs: ("warmup", 10.0))
    monkeypatch.setattr(
        module,
        "timed_prediction_with_telemetry",
        lambda *args, **kwargs: ("Answer", timing),
    )
    monkeypatch.setattr(module, "model_input_device", lambda model, torch: "cpu")

    module.evaluate(tmp_path / "config.yaml", output_dir=output_dir)

    report = (output_dir / "performance_report.json").read_text(encoding="utf-8")
    summary = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert '"completed_samples": 1' in report
    assert '"ttft_ms"' in report
    assert '"performance"' in summary
