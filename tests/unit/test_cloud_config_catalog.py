from pathlib import Path

import yaml

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.quantization.config import load_quantization_config
from sat_rs_vlm.training.config import load_training_config

PROJECT_ROOT = Path(__file__).parents[2]
CLOUD_ENV = {
    "PROJECT_ROOT": "/workspace/sat-rs-vlm",
    "DATA_ROOT": "/workspace/data",
    "MODEL_ROOT": "/workspace/models",
    "OUTPUT_ROOT": "/workspace/outputs",
    "CACHE_ROOT": "/workspace/cache",
    "LOCAL_MODEL_DIR": "/workspace/models/Qwen3-VL-2B-Instruct",
    "TRAIN_JSONL": "/workspace/data/train.jsonl",
    "VAL_JSONL": "/workspace/data/validation.jsonl",
    "ADAPTER_PATH": "/workspace/outputs/adapter",
    "REPLAY_ADAPTER_DIR": "/workspace/outputs/replay-generalist-adapter",
}


def test_all_yaml_configs_parse_and_expand_with_cloud_environment() -> None:
    for path in sorted((PROJECT_ROOT / "configs").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        expanded = expand_environment(payload, environ=CLOUD_ENV, allow_unresolved=False)
        assert "${" not in repr(expanded), path


def test_training_and_quantization_configs_validate_with_cloud_paths(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    for name, value in CLOUD_ENV.items():
        monkeypatch.setenv(name, value)
    for path in sorted((PROJECT_ROOT / "configs/train").glob("*.yaml")):
        load_training_config(path)
    for path in sorted((PROJECT_ROOT / "configs/quantization").glob("*.yaml")):
        load_quantization_config(path, environ=CLOUD_ENV)
