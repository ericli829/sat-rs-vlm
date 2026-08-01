import pytest

from sat_rs_vlm.configuration.environment import expand_environment


def test_environment_expansion_is_recursive() -> None:
    value = {"paths": ["${DATA_ROOT}/images", {"model": "${MODEL_ROOT}"}]}
    expanded = expand_environment(
        value,
        environ={"DATA_ROOT": "/data", "MODEL_ROOT": "/models"},
    )
    assert expanded == {"paths": ["/data/images", {"model": "/models"}]}


def test_missing_environment_variable_is_explicit() -> None:
    with pytest.raises(ValueError, match="MISSING"):
        expand_environment("${MISSING}/data", environ={})


def test_missing_variable_can_be_preserved_for_cloud_config_inspection() -> None:
    assert (
        expand_environment("${REMOTE_ROOT}/data", environ={}, allow_unresolved=True)
        == "${REMOTE_ROOT}/data"
    )
