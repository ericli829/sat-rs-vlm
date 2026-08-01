from pathlib import Path

import pytest

from sat_rs_vlm.configuration.paths import PathConfig, resolve_path_value


def test_relative_paths_resolve_against_project_root(tmp_path: Path) -> None:
    paths = PathConfig.from_mapping(
        {"output_root": "out", "cache_root": ".cache"},
        project_root=tmp_path,
        environ={},
    )
    assert paths.output_root == (tmp_path / "out").resolve()
    assert paths.cache_root == (tmp_path / ".cache").resolve()


def test_environment_path_overrides_yaml(tmp_path: Path) -> None:
    data = tmp_path / "external-data"
    paths = PathConfig.from_mapping(
        {"dataset_root": "yaml-data"},
        project_root=tmp_path,
        environ={"DATA_ROOT": str(data)},
    )
    assert paths.dataset_root == data.resolve()


def test_input_directories_are_not_created_silently(tmp_path: Path) -> None:
    missing = tmp_path / "missing-data"
    paths = PathConfig.from_mapping(
        {"dataset_root": str(missing), "model_root": str(tmp_path / "missing-model")},
        project_root=tmp_path,
        environ={},
    )
    with pytest.raises(FileNotFoundError, match="dataset_root"):
        paths.validate_inputs()
    assert not missing.exists()


def test_output_directories_can_be_created(tmp_path: Path) -> None:
    paths = PathConfig.from_mapping({}, project_root=tmp_path, environ={})
    paths.create_output_directories()
    assert paths.output_root.is_dir()
    assert paths.hf_hub_cache.is_dir()


def test_windows_style_path_is_accepted(tmp_path: Path) -> None:
    resolved = resolve_path_value(r"C:\datasets\VRSBench", base_dir=tmp_path)
    assert str(resolved).lower().startswith("c:")
