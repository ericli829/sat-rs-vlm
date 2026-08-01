"""统一的遥感 VLM 分任务评测指标。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sat_rs_vlm.data.task_protocol import parse_count, parse_detection


def normalize_text(text: str) -> str:
    """小写、去引号和标点，并压缩空白。"""

    value = re.sub(r"[\"'`]", "", text.strip().lower())
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def keyword_hit(prediction: str, reference: str) -> bool:
    """仅用于诊断的关键词命中，不作为主要准确率。"""

    tokens = set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", reference.lower()))
    return bool(tokens) and any(token in prediction.lower() for token in tokens)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """计算 normalized xyxy 轴对齐框 IoU。"""

    ax1, ay1, ax2, ay2 = (float(item) for item in box_a)
    bx1, by1, bx2, by2 = (float(item) for item in box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


def _ngrams(tokens: Sequence[str], size: int) -> Counter[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return Counter()
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _bleu_precision(prediction: str, reference: str, size: int) -> float:
    predicted = _ngrams(_tokens(prediction), size)
    expected = _ngrams(_tokens(reference), size)
    total = sum(predicted.values())
    overlap = sum(min(count, expected[gram]) for gram, count in predicted.items())
    return overlap / total if total else 0.0


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def score_detection(prediction: str, reference: str) -> dict[str, Any]:
    """分别统计 JSON、坐标范围、标签、IoU 和联合正确率。"""

    predicted = parse_detection(prediction)
    expected = parse_detection(reference)
    valid_json = predicted is not None
    coordinate_valid = bool(predicted and predicted.valid_coordinate_range)
    reference_valid = bool(expected and expected.valid_coordinate_range)
    comparable = coordinate_valid and reference_valid
    if comparable and predicted is not None and expected is not None:
        iou = box_iou(predicted.bbox, expected.bbox)
        label_match = predicted.label == expected.label
    else:
        iou = None
        label_match = False
    return {
        "valid_json": valid_json,
        "valid_coordinate_range": coordinate_valid,
        "reference_valid": reference_valid,
        "label_exact_match": label_match,
        "iou": iou,
        "iou_at_0_5": float(iou >= 0.5) if iou is not None else 0.0,
        "detection_exact_at_0_5": float(label_match and iou is not None and iou >= 0.5),
    }


def score_counting(prediction: str, reference: str) -> dict[str, Any]:
    """统计计数可解析率、MAE、精确准确率和 ±1 准确率。"""

    predicted = parse_count(prediction)
    expected = parse_count(reference)
    predicted_value = predicted.value
    expected_value = expected.value
    if predicted_value is None or expected_value is None:
        return {"parsable": False, "mae": None, "acc_exact": 0.0, "acc_within_1": 0.0}
    error = abs(predicted_value - expected_value)
    return {
        "parsable": True,
        "mae": float(error),
        "acc_exact": float(error == 0),
        "acc_within_1": float(error <= 1),
    }


def score_caption(prediction: str, reference: str) -> dict[str, float]:
    """提供无额外依赖的 BLEU-1、BLEU-4 与 ROUGE-L 近似指标。"""

    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    rouge_l = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bleu_1 = _bleu_precision(prediction, reference, 1)
    bleu_4 = _bleu_precision(prediction, reference, 4)
    return {"bleu": bleu_1, "bleu_1": bleu_1, "bleu_4": bleu_4, "rouge_l": rouge_l}


def score_text_task(prediction: str, reference: str) -> dict[str, float]:
    """VQA 与场景分类的精确和归一化精确匹配。"""

    return {
        "exact_match": float(prediction.strip() == reference.strip()),
        "normalized_exact_match": float(normalize_text(prediction) == normalize_text(reference)),
        "keyword_hit": float(keyword_hit(prediction, reference)),
    }


def score_sample(task_type: str, prediction: str, reference: str) -> dict[str, Any]:
    """按任务类型分发单条指标。"""

    task = task_type.strip().lower()
    if task == "detection":
        return score_detection(prediction, reference)
    if task == "counting":
        return score_counting(prediction, reference)
    if task == "captioning":
        return score_caption(prediction, reference)
    return score_text_task(prediction, reference)


def _mean(values: Iterable[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def summarize_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """生成 overall/by_task 稳定摘要，缺失值保留为 None。"""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("task_type", "unknown"))].append(row)
    empty = sum(not str(row.get("prediction", "")).strip() for row in predictions)
    latency = [
        float(row["inference_latency_ms"])
        for row in predictions
        if row.get("inference_latency_ms") is not None
    ]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "metrics_version": "v2_task_metrics",
        "overall": {
            "num_samples": len(predictions),
            "empty_predictions": empty,
            "empty_prediction_rate": empty / len(predictions) if predictions else None,
            "inference_latency_ms": _mean(latency),
        },
        "by_task": {},
    }
    for task, rows in sorted(grouped.items()):
        scores = [
            score_sample(task, str(row.get("prediction", "")), str(row.get("reference", "")))
            for row in rows
        ]
        task_metrics: dict[str, Any] = {
            "num_samples": len(rows),
            "empty_prediction_rate": sum(
                not str(row.get("prediction", "")).strip() for row in rows
            )
            / len(rows),
            "average_generation_length": _mean(
                len(str(row.get("prediction", ""))) for row in rows
            ),
        }
        keys = sorted({key for score in scores for key in score})
        metric_names = {
            "valid_json": "valid_json_rate",
            "parsable": "parsable_rate",
            "iou": "mean_iou",
        }
        for key in keys:
            values = [score.get(key) for score in scores]
            numeric = [value for value in values if isinstance(value, (int, float, bool))]
            task_metrics[metric_names.get(key, key)] = _mean(numeric)
        summary["by_task"][task] = task_metrics
    return summary
