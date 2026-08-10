"""无第三方依赖的内部评测指标。"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from sat_rs_vlm.evaluation.parsers import normalize_text, text_tokens


def metric_value(
    value: float | None,
    *,
    num_samples: int,
    status: str = "ok",
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "label": "internal",
        "status": status,
        "num_samples": num_samples,
        "note": note,
    }


def mean(values: Iterable[float | int | bool | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(item) for item in box_a)
    bx1, by1, bx2, by2 = (float(item) for item in box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def generalized_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Generalized IoU for two axis-aligned boxes."""

    ax1, ay1, ax2, ay2 = (float(item) for item in box_a)
    bx1, by1, bx2, by2 = (float(item) for item in box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    enclosing = max(0.0, max(ax2, bx2) - min(ax1, bx1)) * max(0.0, max(ay2, by2) - min(ay1, by1))
    return iou - (enclosing - union) / enclosing if enclosing > 0 else iou


def normalized_center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Center distance divided by the unit-square diagonal."""

    ax1, ay1, ax2, ay2 = (float(item) for item in box_a)
    bx1, by1, bx2, by2 = (float(item) for item in box_b)
    dx = (ax1 + ax2 - bx1 - bx2) / 2
    dy = (ay1 + ay2 - by1 - by2) / 2
    return math.hypot(dx, dy) / math.sqrt(2.0)


def keyword_hit(prediction: str, reference: str) -> bool:
    tokens = set(text_tokens(reference))
    prediction_lower = prediction.lower()
    return bool(tokens) and any(token in prediction_lower for token in tokens)


def text_task_scores(prediction: str, reference: str) -> dict[str, float | bool]:
    predicted_tokens = text_tokens(prediction)
    reference_tokens = text_tokens(reference)
    predicted_counts = Counter(predicted_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(count, reference_counts[token]) for token, count in predicted_counts.items())
    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normalized_prediction = normalize_text(prediction)
    normalized_reference = normalize_text(reference)
    edit_denominator = max(len(normalized_prediction), len(normalized_reference))
    edit_similarity = (
        1.0
        if edit_denominator == 0
        else 1.0
        - _levenshtein_distance(normalized_prediction, normalized_reference) / edit_denominator
    )
    return {
        "exact_match": prediction.strip() == reference.strip(),
        "normalized_exact_match": normalized_prediction == normalized_reference,
        "keyword_hit": keyword_hit(prediction, reference),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": token_f1,
        "normalized_edit_similarity": edit_similarity,
    }


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _ngrams(tokens: Sequence[str], size: int) -> Counter[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return Counter()
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def bleu_precision_approx(prediction: str, reference: str, size: int) -> float:
    predicted = _ngrams(text_tokens(prediction), size)
    expected = _ngrams(text_tokens(reference), size)
    total = sum(predicted.values())
    overlap = sum(min(count, expected[gram]) for gram, count in predicted.items())
    return overlap / total if total else 0.0


def corpus_bleu_single_reference_approx(
    predictions: Sequence[str], references: Sequence[str], max_order: int
) -> float:
    """Corpus BLEU-N with clipped counts and brevity penalty, using the local tokenizer."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    matches = [0] * max_order
    possible = [0] * max_order
    predicted_length = 0
    reference_length = 0
    for prediction, reference in zip(predictions, references, strict=True):
        predicted_tokens = text_tokens(prediction)
        reference_tokens = text_tokens(reference)
        predicted_length += len(predicted_tokens)
        reference_length += len(reference_tokens)
        for order in range(1, max_order + 1):
            predicted_ngrams = _ngrams(predicted_tokens, order)
            reference_ngrams = _ngrams(reference_tokens, order)
            matches[order - 1] += sum(
                min(count, reference_ngrams[gram]) for gram, count in predicted_ngrams.items()
            )
            possible[order - 1] += sum(predicted_ngrams.values())
    if not predicted_length:
        return 0.0
    precisions = [
        matched / denominator if denominator else 0.0
        for matched, denominator in zip(matches, possible, strict=True)
    ]
    if any(precision == 0.0 for precision in precisions):
        return 0.0
    brevity_penalty = (
        1.0
        if predicted_length > reference_length
        else math.exp(1.0 - reference_length / predicted_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )


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


def rouge_l_f1_approx(prediction: str, reference: str) -> float:
    predicted = text_tokens(prediction)
    expected = text_tokens(reference)
    lcs = _lcs_length(predicted, expected)
    precision = lcs / len(predicted) if predicted else 0.0
    recall = lcs / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def meteor_exact_approx(prediction: str, reference: str) -> float:
    """Dependency-free METEOR approximation using exact unigram matches only."""

    predicted = text_tokens(prediction)
    expected = text_tokens(reference)
    if not predicted or not expected:
        return 0.0
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(expected):
        positions.setdefault(token, []).append(index)
    used: set[int] = set()
    aligned: list[int] = []
    for token in predicted:
        match = next((index for index in positions.get(token, []) if index not in used), None)
        if match is not None:
            used.add(match)
            aligned.append(match)
    matches = len(aligned)
    if not matches:
        return 0.0
    precision = matches / len(predicted)
    recall = matches / len(expected)
    harmonic = 10 * precision * recall / (recall + 9 * precision)
    chunks = 1 + sum(
        current != previous + 1 for previous, current in zip(aligned, aligned[1:], strict=False)
    )
    penalty = 0.5 * (chunks / matches) ** 3
    return harmonic * (1 - penalty)


def chrf_approx(prediction: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score averaged across orders 1..6."""

    predicted = normalize_text(prediction).replace(" ", "")
    expected = normalize_text(reference).replace(" ", "")
    scores: list[float] = []
    beta_squared = beta * beta
    for order in range(1, max_order + 1):
        predicted_ngrams = _ngrams(list(predicted), order)
        expected_ngrams = _ngrams(list(expected), order)
        predicted_total = sum(predicted_ngrams.values())
        expected_total = sum(expected_ngrams.values())
        if not predicted_total or not expected_total:
            scores.append(0.0)
            continue
        overlap = sum(min(count, expected_ngrams[gram]) for gram, count in predicted_ngrams.items())
        precision = overlap / predicted_total
        recall = overlap / expected_total
        denominator = beta_squared * precision + recall
        scores.append((1 + beta_squared) * precision * recall / denominator if denominator else 0.0)
    return statistics.fmean(scores) if scores else 0.0


def cider_d_single_reference_approx_scores(
    predictions: Sequence[str], references: Sequence[str], sigma: float = 6.0
) -> list[float]:
    """Single-reference CIDEr-D approximation with corpus reference IDF."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    document_count = len(references)
    if not document_count:
        return []
    reference_tokens = [text_tokens(reference) for reference in references]
    document_frequencies: dict[int, Counter[tuple[str, ...]]] = {}
    for order in range(1, 5):
        frequency: Counter[tuple[str, ...]] = Counter()
        for tokens in reference_tokens:
            frequency.update(set(_ngrams(tokens, order)))
        document_frequencies[order] = frequency

    def vector(tokens: Sequence[str], order: int) -> tuple[dict[tuple[str, ...], float], float]:
        counts = _ngrams(tokens, order)
        total = sum(counts.values())
        values: dict[tuple[str, ...], float] = {}
        for gram, count in counts.items():
            df = document_frequencies[order].get(gram, 0)
            idf = math.log((document_count + 1.0) / (df + 1.0)) + 1.0
            values[gram] = (count / total) * idf if total else 0.0
        norm = math.sqrt(sum(value * value for value in values.values()))
        return values, norm

    scores: list[float] = []
    for prediction, _reference, expected_tokens in zip(
        predictions, references, reference_tokens, strict=True
    ):
        predicted_tokens = text_tokens(prediction)
        order_scores: list[float] = []
        length_penalty = math.exp(
            -((len(predicted_tokens) - len(expected_tokens)) ** 2) / (2 * sigma * sigma)
        )
        for order in range(1, 5):
            predicted_vector, predicted_norm = vector(predicted_tokens, order)
            expected_vector, expected_norm = vector(expected_tokens, order)
            denominator = predicted_norm * expected_norm
            cosine = (
                sum(
                    value * expected_vector.get(gram, 0.0)
                    for gram, value in predicted_vector.items()
                )
                / denominator
                if denominator
                else 0.0
            )
            order_scores.append(cosine * length_penalty)
        scores.append(10.0 * statistics.fmean(order_scores))
    return scores


def caption_scores(prediction: str, reference: str) -> dict[str, float | int | None]:
    predicted_tokens = text_tokens(prediction)
    reference_tokens = text_tokens(reference)
    return {
        "bleu_1_approx": bleu_precision_approx(prediction, reference, 1),
        "bleu_2_approx": bleu_precision_approx(prediction, reference, 2),
        "bleu_3_approx": bleu_precision_approx(prediction, reference, 3),
        "bleu_4_approx": bleu_precision_approx(prediction, reference, 4),
        "rouge_l_f1_approx": rouge_l_f1_approx(prediction, reference),
        "meteor_exact_approx": meteor_exact_approx(prediction, reference),
        "chrf_approx": chrf_approx(prediction, reference),
        "prediction_token_count": len(predicted_tokens),
        "reference_token_count": len(reference_tokens),
        "length_ratio": (
            len(predicted_tokens) / len(reference_tokens) if reference_tokens else None
        ),
    }


def latency_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    """复用当前仓库量化报告的 nearest-rank P95 规则。"""

    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
            "samples": 0,
        }
    ordered = sorted(float(value) for value in values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
        "samples": len(ordered),
    }
