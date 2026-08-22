from copy import deepcopy
from pathlib import Path

import yaml

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
