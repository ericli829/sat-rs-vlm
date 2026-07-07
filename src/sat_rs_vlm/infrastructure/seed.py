"""随机种子工具。

作用：
    统一设置 Python 标准库随机数种子。后续接入 numpy/torch 时可在此扩展。
"""

import random


def seed_everything(seed: int) -> None:
    """设置随机种子。

    参数：
        seed：整数随机种子。

    返回值：
        None。
    """

    random.seed(seed)
