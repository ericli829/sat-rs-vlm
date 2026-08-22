"""固定评测层级的公共定义。

正式模型提交评测默认使用 E2。E1/E3 只能通过显式配置选择，避免脚本在
不同机器或不同运行时随机截取 validation JSONL，导致结果不可复现。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TIER_NAMES = ("E1", "E2", "E3")
DEFAULT_EVALUATION_TIER = "E2"
LEGACY_TIER_VERSION = "legacy-vrs-v1"
UNIFIED_TIER_VERSION = "unified-v2"
DEFAULT_EVALUATION_TIER_VERSION = UNIFIED_TIER_VERSION
LEGACY_TIER_FILES = {
    "E1": "data/evaluation/tiers/e1_quick.jsonl",
    "E2": "data/evaluation/tiers/e2_standard.jsonl",
    "E3": "data/evaluation/tiers/e3_full.jsonl",
}
UNIFIED_TIER_FILES = {
    "E1": "data/evaluation/tiers_v2/e1_quick.jsonl",
    "E2": "data/evaluation/tiers_v2/e2_standard.jsonl",
    "E3": "data/evaluation/tiers_v2/e3_full.jsonl",
}
DEFAULT_TIER_FILES = UNIFIED_TIER_FILES
LEGACY_TIERS_MANIFEST = "data/evaluation/tiers/evaluation_tiers_manifest.json"
UNIFIED_TIERS_MANIFEST = "data/evaluation/tiers_v2/evaluation_tiers_manifest.json"
DEFAULT_TIERS_MANIFEST = UNIFIED_TIERS_MANIFEST
COUNTING_FOCUSED_TIER = "E_COUNT_V1"
COUNTING_FOCUSED_TIER_FILE = "data/evaluation/tiers/e_count_v1.jsonl"
COUNTING_FOCUSED_TIER_MANIFEST = "data/evaluation/tiers/e_count_v1_manifest.json"


def normalize_tier(value: str | None) -> str:
    """规范化评测层级并拒绝未知值。"""

    tier = str(value or DEFAULT_EVALUATION_TIER).upper()
    if tier not in TIER_NAMES:
        raise ValueError(f"Unknown evaluation tier {value!r}; choose one of {TIER_NAMES}.")
    return tier


def default_tier_file(
    tier: str = DEFAULT_EVALUATION_TIER,
    *,
    tier_version: str = DEFAULT_EVALUATION_TIER_VERSION,
) -> str:
    """返回仓库相对路径形式的固定评测 JSONL 路径。"""

    files = LEGACY_TIER_FILES if tier_version == LEGACY_TIER_VERSION else UNIFIED_TIER_FILES
    if tier_version not in {LEGACY_TIER_VERSION, UNIFIED_TIER_VERSION}:
        raise ValueError(
            f"Unknown evaluation tier version {tier_version!r}; choose "
            f"{LEGACY_TIER_VERSION!r} or {UNIFIED_TIER_VERSION!r}."
        )
    return files[normalize_tier(tier)]


def resolve_tier_identity(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """从评测配置解析 tier、固定样本文件和 tier manifest。

    配置可以显式提供 ``evaluation.tier``、``data.eval_file`` 和
    ``evaluation.tiers_manifest``。当未提供 tier 时默认 E2；当未提供评测
    文件时才使用标准 E2 路径。显式传入旧数据文件不会被静默改写，调用方
    可以据此给出清晰的兼容性错误。
    """

    evaluation = dict(config.get("evaluation", {}))
    data = dict(config.get("data", {}))
    tier = normalize_tier(evaluation.get("tier"))
    tier_version = str(
        evaluation.get("tier_version") or DEFAULT_EVALUATION_TIER_VERSION
    )
    eval_file = str(
        data.get("eval_file") or default_tier_file(tier, tier_version=tier_version)
    )
    manifest = str(
        evaluation.get("tiers_manifest")
        or data.get("tiers_manifest")
        or (
            LEGACY_TIERS_MANIFEST
            if tier_version == LEGACY_TIER_VERSION
            else UNIFIED_TIERS_MANIFEST
        )
    )
    eval_path = Path(eval_file).expanduser()
    if not eval_path.is_absolute():
        eval_path = project_root / eval_path
    manifest_path = Path(manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    return {
        "tier": tier,
        "tier_version": tier_version,
        "eval_file": eval_file,
        "eval_path": eval_path,
        "tiers_manifest": manifest,
        "tiers_manifest_path": manifest_path,
        "is_default_tier_file": eval_file.replace("\\", "/")
        == default_tier_file(tier, tier_version=tier_version),
    }


def load_tier_record(manifest_path: Path, tier: str) -> dict[str, Any] | None:
    """读取 manifest 中的层级记录；manifest 缺失时返回 ``None``。

    读取失败由调用方根据 ``strict`` 策略处理。该函数保持轻量，方便单元
    测试和不加载模型的提交前配置检查复用。
    """

    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError(f"Tier manifest has no tiers mapping: {manifest_path}")
    record = tiers.get(normalize_tier(tier))
    return dict(record) if isinstance(record, dict) else None


def validate_tier_asset(
    *,
    tier: str,
    eval_file: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """校验固定 JSONL 存在且与 tier manifest 中的 SHA256 一致。"""

    normalized = normalize_tier(tier)
    if not eval_file.is_file():
        raise FileNotFoundError(
            f"Evaluation tier {normalized} file is missing: {eval_file}. "
            "Run scripts/evaluation/build_evaluation_tiers.py first."
        )
    record = load_tier_record(manifest_path, normalized)
    if record is None:
        raise FileNotFoundError(
            f"Evaluation tier {normalized} is not recorded in manifest: {manifest_path}"
        )
    actual_hash = file_sha256(eval_file)
    expected_hash = record.get("sha256")
    canonical_hash = actual_hash
    if expected_hash and str(expected_hash) != actual_hash and eval_file.suffix.lower() == ".jsonl":
        canonical_hash = canonical_jsonl_sha256(eval_file)
    if expected_hash and str(expected_hash) != canonical_hash:
        raise ValueError(
            f"Evaluation tier {normalized} SHA256 mismatch: expected {expected_hash}, "
            f"got raw={actual_hash}, canonical_jsonl={canonical_hash} for {eval_file}"
        )
    expected_count = record.get("sample_count")
    if expected_count is not None:
        actual_count = sum(
            1
            for line in eval_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if int(expected_count) != actual_count:
            raise ValueError(
                f"Evaluation tier {normalized} sample count mismatch: "
                f"expected {expected_count}, got {actual_count}"
            )
    return {
        "tier": normalized,
        "tier_version": str(
            json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "tier_version", LEGACY_TIER_VERSION
            )
        ),
        "sha256": canonical_hash,
        "raw_sha256": actual_hash,
        "sample_count": expected_count,
    }


def file_sha256(path: Path) -> str:
    """计算 tier 文件 SHA256，供评测 manifest 记录。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_jsonl_sha256(path: Path) -> str:
    """计算以 LF 为 canonical 换行的 JSONL hash，兼容 Windows CRLF checkout。"""

    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()
