from copy import deepcopy
from pathlib import Path

import yaml

from sat_rs_vlm.models.rs_merger_expert import rs_detail_parameter_count

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiments" / name).read_text(encoding="utf-8"))


def test_c4_is_c2_with_only_width_and_identity_changed():
    c2 = deepcopy(_load("rs_count_merger_c2_detail_4090.yaml"))
    c4 = deepcopy(_load("rs_count_merger_c4_wide_4090.yaml"))
    assert c2["expert"]["detail_hidden_size"] == 512
    assert c4["expert"]["detail_hidden_size"] == 1024
    assert c4["expert"]["expert_variant"] == "rs_detail"
    assert c4["expert"]["interface_lora"]["enabled"] is False
    c2["experiment"] = c4["experiment"] = "experiment"
    c2["expert"]["variant"] = c4["expert"]["variant"] = "variant"
    c2["expert"]["detail_hidden_size"] = c4["expert"]["detail_hidden_size"] = 0
    assert c2 == c4


def test_continuation_configs_lock_requested_extra_epoch_and_lrs():
    c2 = _load("rs_count_merger_c2_cont_4090.yaml")
    c3 = _load("rs_count_merger_c3_cont_4090.yaml")
    assert c2["continuation"]["source_checkpoint"] == "${C2_EXPERT_CHECKPOINT}"
    assert c3["continuation"]["source_checkpoint"] == "${C3_EXPERT_CHECKPOINT}"
    assert c2["training"]["target_effective_epochs"] == 0.5
    assert c3["training"]["target_effective_epochs"] == 0.5
    assert c2["training"]["merger_lr"] == 5e-5
    assert c3["training"]["merger_lr"] == 5e-5
    assert c3["training"]["interface_lora_lr"] == 1e-5
    assert c2["training"]["max_grad_norm"] == 1.0
    assert c3["training"]["max_grad_norm"] == 1.0
    assert c2["model"]["r1_integration"] == "additive"
    assert c3["model"]["r1_integration"] == "additive"
    assert c2["provenance"]["architecture_audit"] == "${SOURCE_ARCHITECTURE_AUDIT}"
    assert c3["provenance"]["architecture_audit"] == "${SOURCE_ARCHITECTURE_AUDIT}"


def test_four_epoch_lm_and_count_matrix_keeps_only_requested_objective_difference():
    for prefix in ("c2", "c3"):
        lm = _load(f"rs_count_merger_{prefix}_lm_4e.yaml")
        count = _load(f"rs_count_merger_{prefix}_count_4e.yaml")
        assert lm["training"]["target_effective_epochs"] == 4.0
        assert count["training"]["target_effective_epochs"] == 4.0
        assert lm["training"]["warmup_ratio"] == 0.05
        assert count["training"]["scheduler"] == "cosine"
        assert lm["training"]["count_loss"]["enabled"] is False
        assert count["training"]["count_loss"]["enabled"] is True
        assert count["training"]["count_loss"]["distribution"] == "categorical"
        assert count["training"]["count_loss"]["feature_layer"] == 3
        assert count["training"]["count_loss"]["max_count"] == 15
        assert count["expert"]["local_depth"] == 1


def test_four_epoch_configs_use_external_fixed_eval_assets():
    for path in (ROOT / "configs" / "experiments").glob("rs_count_merger_*_4e.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["data"]["fixed_eval"] == "${E_COUNT_V2_FILE}"
        assert payload["data"]["fixed_eval_manifest"] == "${E_COUNT_V2_MANIFEST}"
        assert payload["data"]["eval_image_root"] == "${EVAL_DATA_ROOT}"


def test_c4_wide_four_epoch_is_pure_c2_width_control():
    c2 = _load("rs_count_merger_c2_lm_4e.yaml")
    c4 = _load("rs_count_merger_c4_wide_count_4e.yaml")
    for key in (
        "bf16",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "target_effective_epochs",
        "gradient_checkpointing",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "max_grad_norm",
        "seed",
        "max_seq_length",
        "merger_lr",
        "interface_lora_lr",
        "merger_weight_decay",
        "warmup_ratio",
        "scheduler",
    ):
        assert c4["training"][key] == c2["training"][key]
    assert c4["training"]["count_loss"] == {"enabled": False}
    assert c4["expert"]["expert_variant"] == "rs_detail"
    assert c4["expert"]["interface_lora"]["enabled"] is False
    assert c4["expert"]["detail_hidden_size"] == 1024
    c2_count = 4 * rs_detail_parameter_count(1024, 2560, 512, local_depth=1, spatial_merge_size=2)
    c4_count = 4 * rs_detail_parameter_count(1024, 2560, 1024, local_depth=1, spatial_merge_size=2)
    assert 24_000_000 <= c2_count <= 25_000_000
    assert 50_000_000 <= c4_count <= 51_000_000
