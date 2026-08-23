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
    "EVAL_TIER_ROOT": "/workspace/outputs/evaluation_tiers/unified_v2",
    "EVAL_DATA_ROOT": "/workspace/data",
    "E_COUNT_V2_FILE": "/workspace/outputs/evaluation_tiers/unified_v2/e_count_v2.jsonl",
    "E_COUNT_V2_MANIFEST": (
        "/workspace/outputs/evaluation_tiers/unified_v2/e_count_v2_manifest.json"
    ),
    "R1_CHECKPOINT": "/workspace/outputs/r1/adapter",
    "R1_VISUAL_SIDECAR": "/workspace/outputs/r1/adapter/visual_trainable_weights.safetensors",
    "SOURCE_ARCHITECTURE_AUDIT": "/workspace/outputs/r1/source_architecture_audit.json",
    "C2_EXPERT_CHECKPOINT": "/workspace/outputs/rs_merger_expert/c2/final.safetensors",
    "C3_EXPERT_CHECKPOINT": "/workspace/outputs/rs_merger_expert/c3/final.safetensors",
    "CACHE_ROOT": "/workspace/cache",
    "LOCAL_MODEL_DIR": "/workspace/models/Qwen3-VL-2B-Instruct",
    "QWEN3VL_4B_MODEL_DIR": "/workspace/models/Qwen3-VL-4B-Instruct",
    "TRAIN_JSONL": "/workspace/data/train.jsonl",
    "VAL_JSONL": "/workspace/data/validation.jsonl",
    "ADAPTER_PATH": "/workspace/outputs/adapter",
    "REPLAY_ADAPTER_DIR": "/workspace/outputs/replay-generalist-adapter",
    "QWEN3VL_4B_REPLAY_ADAPTER_DIR": "/workspace/outputs/qwen3vl-4b-replay-adapter",
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
