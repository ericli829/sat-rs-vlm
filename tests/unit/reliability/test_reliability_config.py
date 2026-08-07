import importlib.util
from pathlib import Path

from sat_rs_vlm.configuration.layered import LayeredConfigRequest, load_layered_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_manifest_builder():
    script = PROJECT_ROOT / "scripts/data/build_reliability_eval_manifest.py"
    spec = importlib.util.spec_from_file_location("build_reliability_eval_manifest", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_smoke_config_expands_environment() -> None:
    environment = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_ROOT": str(PROJECT_ROOT / "tests/fixtures/miniature_dataset"),
        "MODEL_ROOT": str(PROJECT_ROOT / ".models"),
        "OUTPUT_ROOT": str(PROJECT_ROOT / "outputs-test"),
    }
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(
                PROJECT_ROOT / "configs/base/default.yaml",
                PROJECT_ROOT / "configs/reliability/base.yaml",
            ),
            environment_config=PROJECT_ROOT / "configs/local/paths.yaml",
            experiment_config=PROJECT_ROOT / "configs/reliability/local_smoke.yaml",
            project_root=PROJECT_ROOT,
        ),
        environ=environment,
    )

    assert config["experiment"]["execution_mode"] == "smoke_mock"
    assert config["paths"]["output_root"] == environment["OUTPUT_ROOT"]
    assert config["data"]["eval_manifest"].endswith("smoke.jsonl")


def test_cloud_config_loads_without_accessing_cloud_paths() -> None:
    environment = {
        "PROJECT_ROOT": "/workspace/sat-rs-vlm",
        "DATA_ROOT": "/mnt/data",
        "MODEL_ROOT": "/mnt/models",
        "OUTPUT_ROOT": "/mnt/outputs",
    }
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(
                PROJECT_ROOT / "configs/base/default.yaml",
                PROJECT_ROOT / "configs/reliability/base.yaml",
            ),
            environment_config=PROJECT_ROOT / "configs/cloud/autodl.yaml",
            experiment_config=PROJECT_ROOT / "configs/reliability/cloud_smoke.yaml",
            project_root=PROJECT_ROOT,
        ),
        environ=environment,
    )

    assert config["experiment"]["execution_mode"] == "smoke_mock"
    assert config["model"]["base_model"] == "/mnt/models/Qwen3-VL-2B-Instruct"
    assert config["data"]["dataset_root"] == "/mnt/data/VRSBench"


def test_formal_bitflip_config_includes_vrsbench_and_levircc() -> None:
    environment = {
        "PROJECT_ROOT": "/workspace/sat-rs-vlm",
        "DATA_ROOT": "/mnt/data",
        "MODEL_ROOT": "/mnt/models",
        "OUTPUT_ROOT": "/mnt/outputs",
    }
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(
                PROJECT_ROOT / "configs/base/default.yaml",
                PROJECT_ROOT / "configs/reliability/base.yaml",
            ),
            environment_config=PROJECT_ROOT / "configs/cloud/autodl.yaml",
            experiment_config=(
                PROJECT_ROOT / "configs/reliability/experiments/lora_bitflip.yaml"
            ),
            project_root=PROJECT_ROOT,
        ),
        environ=environment,
    )

    sources = config["data"]["reliability_sources"]
    assert config["data"]["dataset_root"] == "/mnt/data"
    assert config["data"]["eval_batch_size"] == 8
    assert [source["name"] for source in sources] == ["VRSBench", "LEVIR-CC"]
    assert sources[1]["task_samples"] == {"change_detection": 20}


def test_manifest_builder_derives_multisource_output_when_field_is_missing() -> None:
    builder = _load_manifest_builder()
    output = builder._output_path(
        {"eval_manifest": "/old/vrsbench/eval.jsonl", "reliability_sources": []},
        Path("/mnt/data"),
        multisource=True,
    )

    assert Path(output) == Path(
        "/mnt/data/project_metadata/reliability/vrsbench_levircc_eval.jsonl"
    )
