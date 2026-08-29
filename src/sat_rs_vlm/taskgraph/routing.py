"""Configurable dataset/task execution-mode policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    DIRECT_VLM = "DIRECT_VLM"
    DIRECT_DETECTION = "DIRECT_DETECTION"
    TASKGRAPH_UHR = "TASKGRAPH_UHR"


def _key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


DEFAULT_POLICIES: dict[str, dict[str, ExecutionMode]] = {
    "vrsbench": {
        "caption": ExecutionMode.DIRECT_VLM,
        "captioning": ExecutionMode.DIRECT_VLM,
        "vqa": ExecutionMode.DIRECT_VLM,
        "count": ExecutionMode.DIRECT_DETECTION,
        "counting": ExecutionMode.DIRECT_DETECTION,
        "detection": ExecutionMode.DIRECT_DETECTION,
        "default": ExecutionMode.DIRECT_VLM,
    },
    "levir_cc": {
        "change_caption": ExecutionMode.DIRECT_VLM,
        "change_vqa": ExecutionMode.DIRECT_VLM,
        "change_detection": ExecutionMode.DIRECT_VLM,
        "default": ExecutionMode.DIRECT_VLM,
    },
    "mme_realworld_rs": {"default": ExecutionMode.TASKGRAPH_UHR},
    "xlrs_bench": {"default": ExecutionMode.TASKGRAPH_UHR},
}


@dataclass
class DatasetExecutionPolicy:
    policies: dict[str, dict[str, ExecutionMode]] = field(
        default_factory=lambda: {name: dict(values) for name, values in DEFAULT_POLICIES.items()}
    )
    unknown_default: ExecutionMode = ExecutionMode.TASKGRAPH_UHR

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> DatasetExecutionPolicy:
        if not value:
            return cls()
        policies = {name: dict(values) for name, values in DEFAULT_POLICIES.items()}
        for dataset, raw_rules in value.items():
            if not isinstance(raw_rules, dict):
                raise TypeError(f"dataset policy {dataset!r} must be a mapping")
            policies[_key(dataset)] = {
                _key(task): mode if isinstance(mode, ExecutionMode) else ExecutionMode(str(mode))
                for task, mode in raw_rules.items()
            }
        return cls(policies)

    def resolve(self, dataset: str, task_category: str) -> ExecutionMode:
        rules = self.policies.get(_key(dataset))
        if rules is None:
            return self.unknown_default
        return rules.get(_key(task_category), rules.get("default", self.unknown_default))


class ExecutionModeRouter:
    def __init__(self, policy: DatasetExecutionPolicy | None = None) -> None:
        self.policy = policy or DatasetExecutionPolicy()

    def route(self, dataset: str, task_category: str) -> ExecutionMode:
        return self.policy.resolve(dataset, task_category)
