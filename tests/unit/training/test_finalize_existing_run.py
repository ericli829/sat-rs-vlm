from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from scripts.training import train_rs_merger_expert as train_script


class _Controller:
    def __init__(self) -> None:
        self.state = {"expert.weight": torch.zeros(2)}

    def load_expert_state_dict(self, state):
        self.state = {name: value.detach().cpu().clone() for name, value in state.items()}

    def expert_state_dict(self):
        return self.state


def _run(tmp_path: Path, completed: float = 4.0) -> tuple[Path, dict, _Controller]:
    root = tmp_path / "run"
    (root / "epoch_checkpoints").mkdir(parents=True)
    for epoch in range(1, 5):
        epoch_dir = root / "epoch_checkpoints" / f"epoch_{epoch:02d}"
        epoch_dir.mkdir()
        save_file(
            {"expert.weight": torch.full((2,), float(epoch))},
            str(epoch_dir / "expert_model.safetensors"),
        )
        (epoch_dir / "epoch_manifest.json").write_text(
            json.dumps({"epoch": epoch, "completed_effective_epochs": float(epoch)}),
            encoding="utf-8",
        )
    torch.save(
        {"completed_effective_epochs": completed, "global_optimizer_step": 3860},
        root / "training_state.pt",
    )
    (root / "train_log.jsonl").write_text(
        '{"optimizer_step":3860,"loss_total":1.0}\n', encoding="utf-8"
    )
    controller = _Controller()
    config = {"experiment": "C2_LM_4E", "training": {"target_effective_epochs": 4.0}, "data": {}}
    return root, config, controller


def test_continuation_manifest_records_parent_checkpoint_identity(tmp_path: Path) -> None:
    files = {
        "r1": tmp_path / "r1",
        "sidecar": tmp_path / "sidecar.bin",
        "audit": tmp_path / "audit.json",
        "data": tmp_path / "train.jsonl",
        "data_manifest": tmp_path / "data_manifest.json",
    }
    files["r1"].mkdir()
    (files["r1"] / "strategy_manifest.json").write_text("{}", encoding="utf-8")
    for key in ("sidecar", "audit", "data", "data_manifest"):
        files[key].write_text(key, encoding="utf-8")
    config = {
        "experiment": "C2_CONT",
        "model": {"r1_checkpoint": str(files["r1"]), "visual_sidecar": str(files["sidecar"])},
        "data": {"train_file": str(files["data"])},
        "training": {"count_loss": {}},
        "provenance": {"architecture_audit": str(files["audit"])},
    }
    manifest = train_script._build_expert_manifest(
        config=config,
        expert={
            "variant": "c2_rs_detail",
            "expert_variant": "rs_detail",
            "detail_hidden_size": 512,
            "local_depth": 1,
            "interface_lora": {"enabled": False},
        },
        architecture={
            "deepstack_visual_indexes": [2, 5, 8],
            "vision_block_count": 12,
            "spatial_merge_size": 2,
            "vision_hidden_size": 1024,
            "llm_hidden_size": 2560,
        },
        r1_report={"implementation": "merge"},
        audit_sha="audit-sha",
        r1_manifest_sha="r1-sha",
        sidecar_sha="sidecar-sha",
        data_manifest=files["data_manifest"],
        trainable_audit={
            "expert_parameter_count": 24,
            "interface_lora_parameter_count": 0,
            "total_trainable_parameter_count": 24,
            "count_head_parameter_count": 0,
        },
        resume_report={
            "source": "parent/checkpoint",
            "expert_weights_sha256": "weights-sha",
            "expert_manifest_sha256": "manifest-sha",
        },
        transformers=SimpleNamespace(__version__="test"),
        torch=SimpleNamespace(__version__="test"),
        peft=SimpleNamespace(__version__="test"),
    )
    assert manifest["parent_checkpoint"] == {
        "path": "parent/checkpoint",
        "expert_weights_sha256": "weights-sha",
        "expert_manifest_sha256": "manifest-sha",
    }


def test_finalize_existing_run_rejects_mismatched_training_configuration(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    torch.save({"completed_effective_epochs": 4.0}, root / "training_state.pt")
    (root / "preflight.json").write_text(
        json.dumps(
            {
                "experiment": "C2_COUNT_4E",
                "target_effective_epochs": 4.0,
                "training_plan": {"expected_effective_epochs": 4.0},
                "training_config": {"target_effective_epochs": 4.0, "merger_lr": 1e-4},
                "expert_provenance": {
                    "variant": "c2_rs_detail",
                    "expert_variant": "rs_detail",
                    "detail_hidden_size": 512,
                    "local_depth": 1,
                    "interface_lora_enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )
    config = {
        "experiment": "C2_COUNT_4E",
        "training": {"target_effective_epochs": 4.0, "merger_lr": 5e-5},
    }
    with pytest.raises(ValueError, match="training_config"):
        train_script.finalize_existing_run(
            root=root,
            config=config,
            expert={
                "variant": "c2_rs_detail",
                "expert_variant": "rs_detail",
                "detail_hidden_size": 512,
                "local_depth": 1,
                "interface_lora": {"enabled": False},
            },
            model=None,
            processor=None,
            controller=_Controller(),
            manifest={"schema_version": "2.0"},
            torch=torch,
        )


def test_normal_and_recovery_final_state_serialization_is_byte_identical(
    tmp_path: Path, monkeypatch
) -> None:
    normal_root = tmp_path / "normal_checkpoint"
    normal_controller = _Controller()
    normal_controller.state = {"expert.weight": torch.full((2,), 4.0)}
    base_manifest = {
        "schema_version": "2.0",
        "architecture_audit_sha256": "audit",
        "source_r1_manifest_sha256": "r1",
        "source_visual_sidecar_sha256": "sidecar",
    }
    train_script.save_composite_checkpoint(
        normal_controller,
        normal_root,
        manifest=base_manifest,
        training_summary={"mode": "normal"},
        resolved_config={"experiment": "C2_LM_4E"},
    )

    recovery_root, config, recovery_controller = _run(tmp_path / "recovery")
    monkeypatch.setattr(train_script, "_evaluate_fixed_curve", lambda **_: [])
    train_script.finalize_existing_run(
        root=recovery_root,
        config=config,
        expert={"variant": "c2_rs_detail"},
        model=None,
        processor=None,
        controller=recovery_controller,
        manifest=dict(base_manifest),
        torch=torch,
    )
    normal_bytes = (normal_root / "expert_model.safetensors").read_bytes()
    recovery_bytes = (recovery_root / "checkpoint" / "expert_model.safetensors").read_bytes()
    assert normal_bytes == recovery_bytes
    normal_manifest = json.loads((normal_root / "expert_manifest.json").read_text())
    recovery_manifest = json.loads(
        (recovery_root / "checkpoint" / "expert_manifest.json").read_text()
    )
    for key in base_manifest:
        assert recovery_manifest[key] == normal_manifest[key]


def test_finalize_existing_run_writes_epoch04_exact_state_without_training(tmp_path: Path) -> None:
    root, config, controller = _run(tmp_path)
    summary = train_script.finalize_existing_run(
        root=root,
        config=config,
        expert={"variant": "c2_rs_detail"},
        model=None,
        processor=None,
        controller=controller,
        manifest={"schema_version": "2.0"},
        torch=torch,
    )
    assert summary["finalization"]["training_reexecuted"] is False
    assert summary["provenance_validation"]["status"] == "validated_with_unknowns"
    assert "training_config" in summary["provenance_validation"]["unknown_fields"]
    saved = load_file(str(root / "checkpoint" / "expert_model.safetensors"), device="cpu")
    assert torch.equal(saved["expert.weight"], torch.full((2,), 4.0))
    checkpoint_manifest = json.loads(
        (root / "checkpoint" / "expert_manifest.json").read_text()
    )
    assert checkpoint_manifest["finalization"]["source_epoch"] == 4
    assert checkpoint_manifest["provenance_validation"]["status"] == "validated_with_unknowns"


def test_finalize_existing_run_fails_before_finalize_when_incomplete(tmp_path: Path) -> None:
    root, config, controller = _run(tmp_path, completed=3.5)
    with pytest.raises(ValueError, match="completed_effective_epochs"):
        train_script.finalize_existing_run(
            root=root,
            config=config,
            expert={"variant": "c2_rs_detail"},
            model=None,
            processor=None,
            controller=controller,
            manifest={"schema_version": "2.0"},
            torch=torch,
        )
