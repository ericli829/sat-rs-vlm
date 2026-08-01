from pathlib import Path

from sat_rs_vlm.configuration.layered import LayeredConfigRequest, load_layered_config


def _yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_layer_priority_cli_over_environment_over_experiment(tmp_path: Path) -> None:
    base = _yaml(
        tmp_path / "base.yaml", "training:\n  max_steps: 100\npaths:\n  output_root: base\n"
    )
    environment = _yaml(tmp_path / "env.yaml", "training:\n  max_steps: 50\n")
    experiment = _yaml(tmp_path / "experiment.yaml", "training:\n  max_steps: 10\n")
    request = LayeredConfigRequest(
        base_configs=[base],
        environment_config=environment,
        experiment_config=experiment,
        cli_overrides={"training.max_steps": 1},
        project_root=tmp_path,
    )
    config = load_layered_config(request, environ={"OUTPUT_ROOT": "from-env"})
    assert config["training"]["max_steps"] == 1
    assert config["paths"]["output_root"] == "from-env"


def test_cloud_config_can_load_without_accessing_cloud_paths(tmp_path: Path) -> None:
    cloud = _yaml(
        tmp_path / "cloud.yaml",
        "paths:\n  dataset_root: /root/autodl-tmp/datasets\n",
    )
    config = load_layered_config(
        LayeredConfigRequest(environment_config=cloud, project_root=tmp_path),
        environ={},
    )
    assert config["paths"]["dataset_root"] == "/root/autodl-tmp/datasets"
