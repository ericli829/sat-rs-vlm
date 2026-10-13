from __future__ import annotations

import json

import pytest

from sat_rs_vlm.models.qwen3vl_shared import SharedQwen3VLModelManager


def _adapter(path, *, rank=16, modules=None):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen3-VL-4B-Instruct",
                "r": rank,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": modules or ["q_proj", "v_proj"],
                "bias": "none",
                "fan_in_fan_out": False,
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"fixture")


def _manager(tmp_path, *, route_rank=16, route_modules=None):
    base = tmp_path / "Qwen3-VL-4B-Instruct"
    base.mkdir()
    planner = tmp_path / "planner"
    route = tmp_path / "changechat"
    _adapter(planner)
    _adapter(route, rank=route_rank, modules=route_modules)
    return SharedQwen3VLModelManager(
        base_model=str(base),
        processor_source=str(base),
        adapters={"planner": str(planner), "changechat": str(route)},
        model_kwargs={},
        processor_kwargs={},
        default_adapter="planner",
    )


def test_shared_manager_validates_lora_contract_without_loading_models(tmp_path):
    manager = _manager(tmp_path)

    assert manager.loaded is False
    assert manager.metrics["adapter_names"] == ["planner", "changechat"]
    assert manager.metrics["gpu_allocated_mb"] is None


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"route_rank": 8}, "r"),
        ({"route_modules": ["q_proj", "k_proj"]}, "target_modules"),
    ],
)
def test_shared_manager_rejects_incompatible_adapter_shapes(tmp_path, kwargs, field):
    with pytest.raises(ValueError, match=field):
        _manager(tmp_path, **kwargs)


def test_adapter_switch_flushes_role_caches_before_reuse(tmp_path, monkeypatch):
    manager = _manager(tmp_path)

    class Model:
        active = "planner"

        def set_adapter(self, name):
            self.active = name

    model = Model()
    manager._model = model
    manager._processor = object()
    manager._torch = object()
    manager._active_adapter = "planner"
    manager.register_cache_releaser("planner", lambda: released.append("planner"))
    manager.register_cache_releaser("changechat", lambda: released.append("changechat"))
    released = []
    loads = []

    def existing_model():
        loads.append(1)
        return model, manager._processor, object(), manager._torch

    monkeypatch.setattr(manager, "load", existing_model)
    with manager.activate("changechat") as components:
        assert components[0] is model
        assert model.active == "changechat"
        assert released == ["planner", "changechat"]

    assert len(loads) == 1
    assert manager.metrics["adapter_switch_count"] == 1
