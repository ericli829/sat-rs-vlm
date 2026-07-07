"""日志初始化工具。

作用：
    为 CLI、HTTP 或脚本提供统一 logging.basicConfig 配置入口。
"""

import logging


def configure_logging(level: str = "INFO") -> None:
    """配置 Python 标准日志。

    参数：
        level：日志级别字符串，例如 INFO、DEBUG、WARNING。

    返回值：
        None。
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
