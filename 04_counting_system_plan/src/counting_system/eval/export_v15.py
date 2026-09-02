"""导出 sat-rs-vlm Evaluation v1.5 兼容的 predictions JSONL。

对齐 feature/uhr-locator 的 counting_protocol.py 输入字段：
id, task_type, question, prediction, reference, metadata, inference_latency_ms。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAL_COUNTING_PROTOCOL = "formal_e2_parse_count_plus_exact_cardinality_eligibility_v1"
UPSTREAM_BRANCH = "feature/vlm-semantic-alignment"


def row_to_v15_prediction(
    row: Mapping[str, Any],
    *,
    dataset: str = "XLRS-Bench-lite",
    language: str = "en",
    official_protocol: bool = False,
    protocol_mode: str = "detector_counting_global_dedup",
) -> dict[str, Any]:
    """将 COUNT benchmark 行转为 v1.5 runner 可读的单条 prediction。"""
    sample_id = str(row.get("sample_id") or row.get("id") or "")
    pred = row.get("pred")
    ref = row.get("ref")
    latency_sec = float(row.get("latency_sec") or 0.0)
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    fusion = row.get("fusion") if isinstance(row.get("fusion"), dict) else {}
    return {
        "id": sample_id,
        "task_type": "counting",
        "question": str(row.get("question") or ""),
        "prediction": str(pred) if pred is not None else "",
        "reference": str(ref) if ref is not None else "",
        "metadata": {
            "dataset": dataset,
            "language": language,
            "category": row.get("category"),
            "l2_category": row.get("l2_category"),
            "target": row.get("target"),
            "entire": row.get("entire"),
            "region": row.get("region"),
            "source_module": "counting_system",
            "protocol_mode": protocol_mode,
            "official_protocol": official_protocol,
            "metrics_protocol": FORMAL_COUNTING_PROTOCOL,
            "upstream_branch": UPSTREAM_BRANCH,
            "detector": row.get("detector"),
            "choice_match": row.get("choice_match"),
            "answer_letter": row.get("answer_letter"),
        },
        "inference_latency_ms": round(latency_sec * 1000.0, 3),
        "telemetry": {
            "success": pred is not None and ref is not None,
            "route": "count_executor",
            "timing_ms": {
                "detector": round(latency_sec * 1000.0, 3),
                "e2e": round(latency_sec * 1000.0, 3),
            },
            "vision_input": {
                "tile_count": provenance.get("tiles_run") or row.get("detector_calls"),
                "tiles_planned": provenance.get("tiles_planned"),
            },
            "activated_models": [str(row.get("detector") or "unknown")],
            "fusion": fusion or None,
            "raw_count": row.get("raw_count"),
            "detector_calls": row.get("detector_calls"),
        },
    }


def export_predictions_v15(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str = "XLRS-Bench-lite",
    language: str = "en",
    official_protocol: bool = False,
) -> list[dict[str, Any]]:
    return [
        row_to_v15_prediction(
            row,
            dataset=dataset,
            language=language,
            official_protocol=official_protocol,
        )
        for row in rows
    ]


def write_predictions_v15(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str = "XLRS-Bench-lite",
    language: str = "en",
    official_protocol: bool = False,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    exported = export_predictions_v15(
        rows,
        dataset=dataset,
        language=language,
        official_protocol=official_protocol,
    )
    dest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in exported) + "\n",
        encoding="utf-8",
    )
    return dest


def load_benchmark_rows(path: str | Path) -> list[dict[str, Any]]:
    src = Path(path)
    if src.suffix == ".jsonl":
        return [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported benchmark export: {src}")
