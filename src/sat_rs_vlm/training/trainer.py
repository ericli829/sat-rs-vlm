"""支持可替换多任务 loss、采样器与 checkpoint sidecar 的正式 Trainer。"""

from __future__ import annotations

import importlib
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from sat_rs_vlm.training.config import MultitaskLossConfig
from sat_rs_vlm.training.losses import compute_multitask_loss


def create_multitask_trainer_class(transformers: Any) -> type[Any]:
    """基于当前 Transformers 版本创建兼容签名的 ``MultitaskTrainer`` 类。"""

    class MultitaskTrainer(transformers.Trainer):  # type: ignore[misc]
        """在 forward 前剥离 task metadata，并通过独立 Loss Strategy 计算 loss。"""

        def __init__(
            self,
            *args: Any,
            loss_config: MultitaskLossConfig,
            train_sampler: Any | None = None,
            checkpoint_artifact_saver: Callable[[Any, str], None] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.loss_config = loss_config
            self._configured_train_sampler = train_sampler
            self._checkpoint_artifact_saver = checkpoint_artifact_saver
            self._loss_diagnostic_sums: dict[str, float] = defaultdict(float)
            self._loss_diagnostic_counts: dict[str, int] = defaultdict(int)
            self._loss_diagnostic_steps = 0
            self._missing_task_metadata_warned = False

        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            if self._configured_train_sampler is not None:
                return self._configured_train_sampler
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:  # Transformers releases before the optional dataset argument.
                return super()._get_train_sampler()

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            """计算配置化 loss；``task_types`` 永远不会传入模型 ``forward``。"""

            del num_items_in_batch
            model_inputs = dict(inputs)
            task_types = model_inputs.pop("task_types", None)
            labels = model_inputs.pop("labels", None)
            if labels is None:
                raise ValueError("MultitaskTrainer requires assistant-only labels in every batch")
            if task_types is None and not self.loss_config.strict_task_metadata:
                if not self._missing_task_metadata_warned:
                    warnings.warn(
                        "Batch is missing task_types; unknown_task_weight will be used.",
                        stacklevel=2,
                    )
                    self._missing_task_metadata_warned = True
            outputs = model(**model_inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise ValueError("Model output must expose logits for configured multitask loss")
            torch = importlib.import_module("torch")
            result = compute_multitask_loss(
                logits,
                labels,
                task_types,
                self.loss_config,
                torch=torch,
            )
            self._accumulate_loss_diagnostics(result.diagnostics)
            return (result.loss, outputs) if return_outputs else result.loss

        def _accumulate_loss_diagnostics(self, diagnostics: dict[str, Any]) -> None:
            self._loss_diagnostic_sums["loss/total"] += float(diagnostics["loss/total"])
            for task, values in dict(diagnostics["by_task"]).items():
                loss_key = f"loss/task/{task}"
                sample_count = int(values["samples"])
                self._loss_diagnostic_sums[loss_key] += (
                    float(values["mean_sample_loss"]) * sample_count
                )
                self._loss_diagnostic_counts[loss_key] += sample_count
                self._loss_diagnostic_sums[f"supervised_tokens/task/{task}"] += float(
                    values["supervised_tokens"]
                )
                self._loss_diagnostic_sums[f"samples/task/{task}"] += float(values["samples"])
            self._loss_diagnostic_steps += 1

        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
            """按 Trainer logging cadence 写入聚合任务诊断，不逐 step 输出文本。"""

            enriched = dict(logs)
            if self._loss_diagnostic_steps:
                for key, total in self._loss_diagnostic_sums.items():
                    if key == "loss/total":
                        enriched[key] = total / self._loss_diagnostic_steps
                    elif key.startswith("loss/task/"):
                        enriched[key] = total / self._loss_diagnostic_counts[key]
                    else:
                        enriched[key] = total
                self._loss_diagnostic_sums.clear()
                self._loss_diagnostic_counts.clear()
                self._loss_diagnostic_steps = 0
            super().log(enriched, *args, **kwargs)

        def _save(self, output_dir: str | None = None, state_dict: Any | None = None) -> None:
            super()._save(output_dir=output_dir, state_dict=state_dict)
            if self._checkpoint_artifact_saver is not None:
                destination = output_dir or str(self.args.output_dir)
                self._checkpoint_artifact_saver(self.model, destination)

    return MultitaskTrainer


def create_multitask_trainer(
    transformers: Any,
    *,
    loss_config: MultitaskLossConfig,
    train_sampler: Any | None,
    trainer_kwargs: dict[str, Any],
    checkpoint_artifact_saver: Callable[[Any, str], None] | None = None,
) -> Any:
    """统一构造正式多任务 Trainer，避免脚本 monkey patch 或内嵌 loss 数学。"""

    trainer_class = create_multitask_trainer_class(transformers)
    return trainer_class(
        **trainer_kwargs,
        loss_config=loss_config,
        train_sampler=train_sampler,
        checkpoint_artifact_saver=checkpoint_artifact_saver,
    )
