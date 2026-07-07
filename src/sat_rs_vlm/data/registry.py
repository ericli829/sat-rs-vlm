"""数据集注册表。

算法/设计：
    使用内存字典维护 `name -> factory` 映射。这样训练或评测入口只需通过名称
    获取数据集构造函数，不需要硬编码具体数据集类。
"""

from collections.abc import Callable

from sat_rs_vlm.data.base_dataset import BaseRemoteSensingDataset

DatasetFactory = Callable[..., BaseRemoteSensingDataset]


class DatasetRegistry:
    """轻量数据集工厂注册器。

    作用：
        解耦数据集名称和具体构造逻辑，方便插件式接入新数据集。
    """

    def __init__(self) -> None:
        """初始化空注册表。"""

        self._factories: dict[str, DatasetFactory] = {}

    def register(self, name: str, factory: DatasetFactory) -> None:
        """注册数据集工厂。

        参数：
            name：数据集名称，例如 "vrsbench"。
            factory：返回 BaseRemoteSensingDataset 实例的可调用对象。

        返回值：
            None。
        """

        if not name:
            raise ValueError("Dataset name must not be empty.")
        self._factories[name] = factory

    def get(self, name: str) -> DatasetFactory:
        """按名称获取数据集工厂。

        参数：
            name：已注册的数据集名称。

        返回值：
            DatasetFactory：数据集构造函数。

        异常：
            KeyError：名称未注册时抛出，并说明缺失的数据集名称。
        """

        try:
            return self._factories[name]
        except KeyError as exc:
            raise KeyError(f"Dataset is not registered: {name}") from exc
