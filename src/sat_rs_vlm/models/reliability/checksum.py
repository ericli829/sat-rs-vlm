"""文件 checksum 工具。

算法：
    使用 SHA-256 对文件按 1MB 分块流式读取并更新 digest，避免大模型权重文件
    一次性读入内存。后续可用于模型文件完整性校验和星载故障恢复。
"""

from hashlib import sha256
from pathlib import Path


def file_sha256(path: str) -> str:
    """计算文件 SHA-256。

    参数：
        path：待校验文件路径。

    返回值：
        str：十六进制 SHA-256 摘要。
    """

    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
