"""初赛评测协议与资源披露（对齐第一次统一答疑 20260801 + feature/vlm-semantic-alignment）。

COUNT 走检测器 + 全局坐标去重计数，不调用在线 LLM / GPT Judge；
主指标与 sat_rs_vlm.evaluation.counting_protocol 一致。
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..paths import load_config, resolve_existing
from .export_v15 import FORMAL_COUNTING_PROTOCOL, UPSTREAM_BRANCH

# 独立模型参数量（百万），来源：公开模型卡 / 论文；用于 32B 合计申报。
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "grounding_dino_tiny": {
        "display_name": "GroundingDINO-tiny",
        "role": "detector",
        "params_m": 172.0,
        "activated_on_count_path_m": 172.0,
        "hf_id": "IDEA-Research/grounding-dino-tiny",
    },
    "lae_dino_swint": {
        "display_name": "LAE-DINO Swin-T",
        "role": "detector",
        "params_m": 200.0,
        "activated_on_count_path_m": 200.0,
    },
    "georsclip_vit_b32": {
        "display_name": "GeoRSCLIP ViT-B-32",
        "role": "retriever_gate",
        "params_m": 151.0,
        "activated_on_count_path_m": 0.0,  # gate 关闭时不激活
    },
    "bert_base_uncased": {
        "display_name": "bert-base-uncased",
        "role": "detector_text_encoder",
        "params_m": 110.0,
        "activated_on_count_path_m": 110.0,
    },
}

PARAM_LIMIT_B = 32.0


def file_size_mb(path: str | Path | None) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return round(p.stat().st_size / 1024**2, 2)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024**2, 2)


def _torch_env() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["cuda_version"] = torch.version.cuda
            out["cudnn_version"] = torch.backends.cudnn.version()
            out["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def collect_test_environment() -> dict[str, Any]:
    """采集可复现性所需的软硬件环境。"""
    env: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import psutil

        env["cpu_count"] = psutil.cpu_count(logical=True)
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        pass
    env.update(_torch_env())
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=5).strip()
        env["nvidia_driver"] = out.splitlines()[0] if out else ""
    except Exception:
        pass
    return env


def _resolve_model_paths(cfg: Mapping[str, Any]) -> dict[str, Path | None]:
    models = cfg.get("paths", {}).get("models") or {}
    return {
        "lae_dino_weights": resolve_existing(models.get("lae_dino_weights")),
        "bert_dir": resolve_existing(models.get("bert_dir")),
        "georsclip_ckpt": resolve_existing(models.get("georsclip_ckpt")),
    }


def inventory_system_models(
    *,
    config: Mapping[str, Any] | None = None,
    backends_used: Sequence[str] | None = None,
    gate_enabled: bool = False,
) -> dict[str, Any]:
    """完整系统口径 + 典型 COUNT 路径激活参数量。"""
    cfg = dict(config or load_config())
    paths = _resolve_model_paths(cfg)
    backends = {b.lower() for b in (backends_used or [])}
    detector_key = "lae_dino_swint" if any("lae" in b for b in backends) else "grounding_dino_tiny"
    if not backends:
        detector_key = "grounding_dino_tiny"

    active_keys = [detector_key, "bert_base_uncased"]
    if gate_enabled:
        active_keys.append("georsclip_vit_b32")

    models: list[dict[str, Any]] = []
    total_params_m = 0.0
    activated_params_m = 0.0
    storage_mb = 0.0

    for key, meta in MODEL_CATALOG.items():
        size_mb = None
        if key == "lae_dino_swint":
            size_mb = file_size_mb(paths["lae_dino_weights"])
        elif key == "bert_base_uncased":
            size_mb = file_size_mb(paths["bert_dir"])
        elif key == "georsclip_vit_b32":
            size_mb = file_size_mb(paths["georsclip_ckpt"])
        elif key == "grounding_dino_tiny":
            hf = (cfg.get("paths", {}).get("models") or {}).get("grounding_dino")
            size_mb = file_size_mb(cfg.get("paths", {}).get("huggingface", {}).get("home")) if hf else None

        entry = {
            "model_id": key,
            **meta,
            "storage_mb": size_mb,
            "in_system": True,
            "activated_on_run": key in active_keys,
        }
        models.append(entry)
        total_params_m += float(meta["params_m"])
        if key in active_keys:
            activated_params_m += float(meta.get("activated_on_count_path_m") or meta["params_m"])
        if size_mb:
            storage_mb += size_mb

    return {
        "param_limit_b": PARAM_LIMIT_B,
        "total_system_params_m": round(total_params_m, 1),
        "total_system_params_b": round(total_params_m / 1000.0, 3),
        "within_32b_limit": total_params_m / 1000.0 <= PARAM_LIMIT_B,
        "activated_params_m": round(activated_params_m, 1),
        "activated_params_b": round(activated_params_m / 1000.0, 3),
        "known_storage_mb": round(storage_mb, 2) if storage_mb else None,
        "models": models,
        "offline_only": True,
        "no_online_llm_api": True,
        "no_gpt_judge_primary": True,
    }


def build_protocol_manifest(
    *,
    config: Mapping[str, Any] | None = None,
    dataset: str = "XLRS-Bench-lite",
    dataset_root: str | Path | None = None,
    language: str = "en",
    protocol_mode: str = "detection_counting",
    official_aligned: bool = False,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """披露数据集版本、预处理与评分口径，便于与技术报告对齐。"""
    cfg = dict(config or load_config())
    count_cfg = cfg.get("count") or {}
    scale_cfg = cfg.get("scale") or {}
    gate_cfg = cfg.get("gate") or {}
    detector_cfg = cfg.get("detector") or {}

    root = Path(dataset_root) if dataset_root else None
    manifest_path = root / "counting.jsonl" if root else None

    return {
        "qa_reference": "第一次统一答疑 20260801",
        "upstream_branch": UPSTREAM_BRANCH,
        "upstream_locator_config": "configs/locator/uhr_hierarchical.yaml",
        "dataset": dataset,
        "dataset_manifest": str(manifest_path) if manifest_path else None,
        "language": language,
        "task": "counting",
        "protocol_mode": protocol_mode,
        "official_aligned": official_aligned,
        "metrics_protocol": FORMAL_COUNTING_PROTOCOL,
        "primary_metrics": ["exact_match", "rmse"],
        "secondary_metrics": ["mae", "within1_accuracy", "choice_accuracy"],
        "excluded_primary_metrics": ["gpt_judge", "clair"],
        "dedup_policy": "global_coordinate_core_ownership_nms",
        "input_pipeline": {
            "entire_default": bool(count_cfg.get("entire", True)),
            "source_scales": scale_cfg.get("source_scales"),
            "default_source_scale": scale_cfg.get("default_source_scale"),
            "native_tile_size": (scale_cfg.get("native") or {}).get("tile_size"),
            "native_overlap": (scale_cfg.get("native") or {}).get("overlap"),
            "fine_tile_size": (scale_cfg.get("fine") or {}).get("tile_size"),
            "fine_overlap": (scale_cfg.get("fine") or {}).get("overlap"),
            "gate_enabled_default": bool(gate_cfg.get("enabled", False)),
            "score_threshold_default": count_cfg.get("score_threshold"),
            "nms_iou": count_cfg.get("nms_iou"),
        },
        "detector": {
            "backend_default": detector_cfg.get("backend"),
            "box_threshold": detector_cfg.get("box_threshold"),
            "text_threshold": detector_cfg.get("text_threshold"),
            "prompt_profiles": "configs/prompt_profiles.yaml",
        },
        "timing_semantics": {
            "latency_sec": "单样本端到端：图像读取 + tiling + 检测 + 融合 + 计数",
            "cold_start_sec": "首次 CountExecutor 初始化与模型加载",
            "ttft_token_per_sec": "不适用（检测器路径，无自回归解码）",
        },
        "notes": list(notes or []),
    }


def _category_key(category: str, l2: str) -> str:
    text = f"{category} {l2}".lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    for hint in (
        "overall_counting",
        "regional_counting",
        "counting_with_complex_reasoning",
        "counting_with_changing_detection",
    ):
        if hint in text:
            return hint
    return category or l2 or "unknown"


def summarize_by_category(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .metrics import choice_match, summarize_counts

    buckets: dict[str, list[tuple[int | None, int | None]]] = {}
    choice_rows: dict[str, list[bool]] = {}
    for row in rows:
        key = _category_key(str(row.get("category") or ""), str(row.get("l2_category") or ""))
        buckets.setdefault(key, []).append((row.get("pred"), row.get("ref")))
        choice_rows.setdefault(key, []).append(
            bool(row.get("choice_match"))
            if "choice_match" in row
            else choice_match(
                int(row["pred"]) if row.get("pred") is not None else -1,
                row.get("options") or [],
                str(row.get("answer_letter") or ""),
            )
        )
    out: dict[str, Any] = {}
    for key, pairs in sorted(buckets.items()):
        summary = summarize_counts(pairs)
        choices = choice_rows.get(key) or []
        summary["choice_accuracy"] = (sum(choices) / len(choices)) if choices else None
        out[key] = summary
    return out


def build_timing_summary(
    latencies: Sequence[float],
    *,
    elapsed_sec: float,
    cold_start_sec: float | None = None,
    warmup: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    lat = list(latencies)
    if not lat:
        return {
            "elapsed_sec": elapsed_sec,
            "cold_start_sec": cold_start_sec,
            "warmup": warmup,
            "repeats": repeats,
            "num_samples": 0,
        }
    ordered = sorted(lat)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return {
        "elapsed_sec": round(elapsed_sec, 4),
        "cold_start_sec": round(cold_start_sec, 4) if cold_start_sec is not None else None,
        "warmup": warmup,
        "repeats": repeats,
        "num_samples": len(lat),
        "mean_latency_sec": round(sum(lat) / len(lat), 4),
        "p50_latency_sec": round(p50, 4),
        "p95_latency_sec": round(p95, 4),
        "min_latency_sec": round(min(lat), 4),
        "max_latency_sec": round(max(lat), 4),
    }


def build_benchmark_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int | None, int | None]],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    backends_used: Sequence[str] | None = None,
    gate_enabled: bool = False,
    dataset_root: str | Path | None = None,
    language: str = "en",
    official_aligned: bool = False,
    cold_start_sec: float | None = None,
    warmup: int = 0,
) -> dict[str, Any]:
    from .metrics import choice_match, summarize_counts

    cfg = dict(config or load_config())
    choice_hits = [
        bool(r.get("choice_match"))
        if "choice_match" in r
        else choice_match(
            int(r["pred"]) if r.get("pred") is not None else -1,
            r.get("options") or [],
            str(r.get("answer_letter") or ""),
        )
        for r in rows
    ]
    overall = dict(metrics)
    overall["exact_match"] = overall.get("exact_accuracy")
    overall["choice_accuracy"] = (sum(choice_hits) / len(choice_hits)) if choice_hits else None
    latencies = [float(r.get("latency_sec") or 0.0) for r in rows]
    return {
        "protocol": build_protocol_manifest(
            config=cfg,
            dataset_root=dataset_root,
            language=language,
            official_aligned=official_aligned,
            notes=[
                "XLRS-Bench-lite 为从完整 XLRS-Bench 导出的 Counting 子集；"
                "与官方全集口径不同，技术报告须单独标注。",
                "本模块采用 TaskGraph COUNT：image/Region 走 tiled 检测，EntitySet/SelectResult 只做 cardinality。",
                "对外输出 ScalarInt；禁止 per-crop 计数直接相加，也禁止再用 VLM 覆盖检测计数。",
                "predictions_v15.jsonl 可直接送入 sat-rs-vlm evaluate_predictions.py + counting_protocol。",
                "choice_accuracy 仅作 XLRS 多选题形式的辅助对照，主指标仍为 exact_match / rmse。",
            ],
        ),
        "environment": collect_test_environment(),
        "resources": inventory_system_models(
            config=cfg,
            backends_used=backends_used,
            gate_enabled=gate_enabled,
        ),
        "metrics": overall,
        "metrics_official_style": summarize_counts(pairs),
        "by_category": summarize_by_category(rows),
        "timing": build_timing_summary(
            latencies,
            elapsed_sec=float(metrics.get("elapsed_sec") or 0.0),
            cold_start_sec=cold_start_sec,
            warmup=warmup,
        ),
    }
