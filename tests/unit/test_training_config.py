from sat_rs_vlm.training.config import load_training_config


def test_training_config_loads() -> None:
    config = load_training_config("configs/train/qwen3vl_lora.yaml")
    assert config.model.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.training.method == "qlora"
    assert config.training.freeze_vision_encoder is True
    assert config.training.dataloader_num_workers == 0
    assert config.training.dataloader_persistent_workers is False
    assert config.lora.r == 16
    assert config.qlora.load_in_4bit is True
