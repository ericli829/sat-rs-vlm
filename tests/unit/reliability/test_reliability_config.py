from pathlib import Path

from sat_rs_vlm.configuration.layered import LayeredConfigRequest, load_layered_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
