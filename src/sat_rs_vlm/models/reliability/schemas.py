"""可靠性模块共享的数据契约。

这些 Pydantic 模型只描述稳定的输入输出，不导入 PyTorch、safetensors 或绘图库，
因此基础安装也可以读取故障记录、校验报告和恢复报告。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BitFlipRecord(BaseModel):
    """一次 bit 翻转的可复现记录。"""

    target_name: str
    flat_index: int
    byte_index: int
    bit_index: int
    dtype: str
    shape: list[int] = Field(default_factory=list)
    before_value: int | float | str | None = None
    after_value: int | float | str | None = None
    before_bytes: str | None = None
    after_bytes: str | None = None
    seed: int | None = None


class AdapterInjectionReport(BaseModel):
    """safetensors LoRA Adapter 注入与完整性验证结果。"""

    schema_version: str = "1.0"
    source_adapter: str
    fault_adapter: str
    source_sha256_before: str
    source_sha256_after: str
    fault_sha256: str
    source_unchanged: bool
    fault_differs: bool
    reload_verified: bool
    changed_parameters: list[str] = Field(default_factory=list)
    records: list[BitFlipRecord] = Field(default_factory=list)


class ChecksumEntry(BaseModel):
    """checksum manifest 中单个文件的相对路径、大小和 SHA-256。"""

    path: str
    size: int
    sha256: str


class ChecksumManifest(BaseModel):
    """可迁移的目录 checksum manifest；`root` 始终是相对路径。"""

    schema_version: str = "1.0"
    algorithm: Literal["sha256"] = "sha256"
    root: str = "."
    files: list[ChecksumEntry] = Field(default_factory=list)


class ChecksumIssue(BaseModel):
    """一个稳定分类的 checksum 验证问题。"""

    path: str
    code: Literal["missing", "size_mismatch", "hash_mismatch"]
    expected: str | int | None = None
    actual: str | int | None = None


class ChecksumVerificationResult(BaseModel):
    """checksum manifest 的结构化验证结果。"""

    valid: bool
    root: str
    checked_files: int
    issues: list[ChecksumIssue] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """模型输出合法性检查结果，错误和警告均使用稳定错误码。"""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    normalized_output: Any = None


class VoteResult(BaseModel):
    """输出过滤与多数投票的决策记录。"""

    selected: str
    has_majority: bool
    used_fallback: bool
    num_inputs: int
    num_valid_inputs: int
    votes: dict[str, int] = Field(default_factory=dict)
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class WeightClampReport(BaseModel):
    """实验性权重裁剪的变更统计。"""

    experimental: bool = True
    processed_parameters: list[str] = Field(default_factory=list)
    clipped_elements: int = 0
    max_abs_adjustment: float = 0.0


class RecoveryResult(BaseModel):
    """文件级 checksum 备份恢复结果。"""

    success: bool
    target_path: str
    backup_path: str
    expected_sha256: str
    before_sha256: str | None = None
    backup_sha256: str | None = None
    after_sha256: str | None = None
    used_atomic_replace: bool = False
    errors: list[str] = Field(default_factory=list)
