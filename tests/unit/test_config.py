from sat_rs_vlm.infrastructure.config import load_config


def test_load_config() -> None:
    settings = load_config()
    assert settings.app.name == "sat-rs-vlm"
    assert settings.model.backend == "mock"
    assert settings.model.model_id == ""
    assert settings.runtime.enable_profiler is True
