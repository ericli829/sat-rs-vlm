"""遥感数据集抽象接口。

作用：
    为后续接入 VRSBench、MME Real RS、XLRS-bench、LEVIR-CC 等数据集保留统一边界。

设计：
    使用抽象基类规定 `__len__` 和 `__getitem__`，避免训练/评测代码依赖具体数据集实现。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseRemoteSensingDataset(ABC):
    """遥感数据集基类。

    子类职责：
        将样本解析为字典，通常包含 image_path、prompt、label、metadata 等字段。
    """

    @abstractmethod
    def __len__(self) -> int:
        """返回数据集样本数量。

        返回值：
            int：可索引样本总数。
        """

        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取单个样本。

        参数：
            index：样本下标。

        返回值：
            dict[str, Any]：规范化后的样本字典，供训练、评测或推理流水线使用。
        """

        raise NotImplementedError
