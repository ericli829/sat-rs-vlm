from __future__ import annotations

import json
from pathlib import Path

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
    saved = load_file(str(root / "checkpoint" / "expert_model.safetensors"), device="cpu")
    assert torch.equal(saved["expert.weight"], torch.full((2,), 4.0))
    checkpoint_manifest = json.loads(
        (root / "checkpoint" / "expert_manifest.json").read_text()
    )
    assert checkpoint_manifest["finalization"]["source_epoch"] == 4


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
