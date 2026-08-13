from __future__ import annotations

from types import SimpleNamespace

import pytest

from sat_rs_vlm.training.config import MultitaskLossConfig
from sat_rs_vlm.training.trainer import create_multitask_trainer_class

torch = pytest.importorskip("torch")


class FakeBaseTrainer:
    def __init__(self, **kwargs: object) -> None:
        self.model = kwargs.get("model")
        self.args = SimpleNamespace(output_dir="unused")
        self.logged: list[dict[str, float]] = []

    def log(self, logs: dict[str, float], *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.logged.append(logs)

    def _get_train_sampler(self, train_dataset: object | None = None) -> None:
        del train_dataset
        return None

    def _save(self, output_dir: str | None = None, state_dict: object | None = None) -> None:
        del output_dir, state_dict


class FakeModel:
    def __init__(self) -> None:
        self.received_keys: set[str] = set()

    def __call__(self, **inputs: object) -> object:
        self.received_keys = set(inputs)
        input_ids = inputs["input_ids"]
        assert torch.is_tensor(input_ids)
        logits = torch.zeros((*input_ids.shape, 2), requires_grad=True)
        return SimpleNamespace(logits=logits)


def test_trainer_strips_metadata_and_labels_before_model_forward() -> None:
    transformers = SimpleNamespace(Trainer=FakeBaseTrainer)
    trainer_class = create_multitask_trainer_class(transformers)
    model = FakeModel()
    trainer = trainer_class(model=model, loss_config=MultitaskLossConfig())
    inputs = {
        "input_ids": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 1], [-100, 1, 1]]),
        "task_types": ["captioning", "counting"],
    }

    loss = trainer.compute_loss(model, inputs)

    assert torch.isfinite(loss)
    assert model.received_keys == {"input_ids", "attention_mask"}
    trainer.log({"learning_rate": 1e-5})
    assert "loss/task/captioning" in trainer.logged[0]
    assert "samples/task/counting" in trainer.logged[0]


def test_trainer_missing_metadata_uses_unknown_weight_only_when_non_strict() -> None:
    transformers = SimpleNamespace(Trainer=FakeBaseTrainer)
    trainer_class = create_multitask_trainer_class(transformers)
    model = FakeModel()
    trainer = trainer_class(
        model=model,
        loss_config=MultitaskLossConfig(strict_task_metadata=False),
    )
    inputs = {
        "input_ids": torch.tensor([[0, 1, 1]]),
        "labels": torch.tensor([[-100, 1, 1]]),
    }

    with pytest.warns(UserWarning, match="missing task_types"):
        loss = trainer.compute_loss(model, inputs)

    assert torch.isfinite(loss)
    assert "task_types" not in model.received_keys
