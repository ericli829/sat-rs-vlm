"""Analyze paired LEVIR-CC change-detection errors without reading images."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


class ErrorAnalysisError(RuntimeError):
    """Raised when paired error-analysis inputs are incompatible."""


_ADDITION = re.compile(
    r"\b(?:appear|appears|appeared|show(?:s|ed)?\s+up|built|build|constructed|"
    r"construction|added|new|emerged|put\s+up)\b",
    re.IGNORECASE,
)
_REMOVAL = re.compile(
    r"\b(?:disappear|disappears|disappeared|removed|removal|demolished|"
    r"demolition|destroyed|absent|no\s+longer)\b",
    re.IGNORECASE,
)
_TRANSFORMATION = re.compile(
    r"\b(?:replace|replaced|replacement|convert|converted|become|becomes|became|"
    r"rebuild|rebuilt|expand|expanded|extend|extended|enlarge|enlarged|widen|"
    r"widened|narrow|narrowed)\b",
    re.IGNORECASE,
)
_OBJECT_PATTERNS = {
    "building": re.compile(
        r"\b(?:buildings?|houses?|homes?|villas?|structures?|roofs?|"
        r"residential\s+(?:areas?|developments?))\b",
        re.IGNORECASE,
    ),
    "road": re.compile(
        r"\b(?:roads?|paths?|trails?|crossroads?|driveways?|parking\s+(?:lots?|areas?))\b",
        re.IGNORECASE,
    ),
    "vegetation": re.compile(
        r"\b(?:trees?|forest|vegetation|grass|plants?|fields?|farmland|crops?)\b",
        re.IGNORECASE,
    ),
    "water": re.compile(r"\b(?:lakes?|pools?|water\s+pools?|ponds?)\b", re.IGNORECASE),
    "vehicle": re.compile(r"\b(?:vehicles?|cars?|trucks?)\b", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=("png", "svg"), default=("png",))
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ErrorAnalysisError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ErrorAnalysisError(f"{path} line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ErrorAnalysisError(f"{path} line {line_number}: row must be an object")
            rows.append(row)
    return rows


def _index(rows: list[dict[str, Any]], role: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ErrorAnalysisError(f"{role}: row has an invalid id")
        if sample_id in indexed:
            raise ErrorAnalysisError(f"{role}: duplicate id {sample_id}")
        indexed[sample_id] = row
    return indexed


def _change_category(text: str) -> str:
    cues = []
    if _ADDITION.search(text):
        cues.append("addition_or_construction")
    if _REMOVAL.search(text):
        cues.append("removal_or_demolition")
    if _TRANSFORMATION.search(text):
        cues.append("replacement_or_expansion")
    if len(cues) > 1:
        return "mixed_change"
    return cues[0] if cues else "other_or_unresolved"


def _object_category(text: str) -> str:
    objects = [name for name, pattern in _OBJECT_PATTERNS.items() if pattern.search(text)]
    return "+".join(objects) if objects else "other_or_unresolved"


def _prediction_template(text: str) -> str:
    normalized = " ".join(text.lower().strip().split()).strip(" .")
    if normalized == "the scene is the same as before":
        return "scene_same_template"
    if normalized == "no change has occurred":
        return "no_change_occurred_template"
    if "no change" in normalized or "unchanged" in normalized or "same as before" in normalized:
        return "other_no_change_expression"
    return "explicit_or_other_description"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "error_type",
        "paired_outcome",
        "reference_changeflag",
        "predicted_changeflag",
        "binary_prediction_source",
        "prediction_template",
        "reference_change_category",
        "reference_object_category",
        "prediction_object_category",
        "candidate_prediction",
        "reference",
        "baseline_prediction",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_font() -> str:
    from matplotlib import font_manager

    candidates = ("Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            return candidate
    raise ErrorAnalysisError("No supported Chinese font is installed for the error-analysis chart.")


def _plot(summary: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _find_font()
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    colors = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#999999")

    error_counts = summary["candidate_errors"]["by_type"]
    labels = ["漏检（FN）", "误报（FP）"]
    values = [error_counts.get("false_negative", 0), error_counts.get("false_positive", 0)]
    bars = axes[0].bar(labels, values, color=(colors[0], colors[1]))
    axes[0].set_title("回放后模型错误数量")
    axes[0].set_ylabel("样本数")
    axes[0].bar_label(bars)

    categories = summary["false_negative_analysis"]["reference_change_category"]
    ordered = sorted(categories.items(), key=lambda item: (-item[1], item[0]))
    category_labels = {
        "addition_or_construction": "新增/建设",
        "removal_or_demolition": "移除/拆除",
        "replacement_or_expansion": "替换/扩建",
        "mixed_change": "混合变化",
        "other_or_unresolved": "其他/未解析",
    }
    axes[1].barh(
        [category_labels.get(name, name) for name, _ in reversed(ordered)],
        [count for _, count in reversed(ordered)],
        color=colors[2],
    )
    axes[1].set_title("漏检参考描述的变化类型")
    axes[1].set_xlabel("样本数")

    transitions = summary["paired_regressions"]["by_error_type"]
    transition_values = [
        transitions.get("false_negative", 0),
        transitions.get("false_positive", 0),
    ]
    bars = axes[2].bar(labels, transition_values, color=(colors[3], colors[4]))
    axes[2].set_title("基线正确、回放后错误的退化样本")
    axes[2].set_ylabel("样本数")
    axes[2].bar_label(bars)

    for axis in axes:
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("LEVIR-CC 回放后模型错误归因（文本规则初筛）", fontsize=17)
    figure.text(
        0.01,
        0.015,
        "仅依据预测 Caption、参考文本和 changeflag 归因；不替代图像级人工复核。",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    generated = []
    for extension in formats:
        path = output_dir / f"levir_cc_error_attribution.{extension}"
        figure.savefig(path, dpi=180 if extension == "png" else None, bbox_inches="tight")
        generated.append(path.name)
    plt.close(figure)
    return generated


def analyze(
    baseline_dir: Path,
    candidate_dir: Path,
    comparison_dir: Path,
    output_dir: Path,
    formats: list[str],
) -> dict[str, Path]:
    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ErrorAnalysisError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    baseline = _index(_read_jsonl(baseline_dir / "evaluated_predictions.jsonl"), "baseline")
    candidate = _index(_read_jsonl(candidate_dir / "evaluated_predictions.jsonl"), "candidate")
    paired = _index(_read_jsonl(comparison_dir / "paired_comparison.jsonl"), "comparison")
    if baseline.keys() != candidate.keys() or baseline.keys() != paired.keys():
        raise ErrorAnalysisError("baseline, candidate and paired comparison IDs must match")

    errors: list[dict[str, Any]] = []
    for sample_id, row in candidate.items():
        reference_flag = row.get("reference_changeflag")
        predicted_flag = row.get("predicted_changeflag")
        if reference_flag == predicted_flag:
            continue
        if reference_flag == 1 and predicted_flag == 0:
            error_type = "false_negative"
        elif reference_flag == 0 and predicted_flag == 1:
            error_type = "false_positive"
        else:
            error_type = "unresolved"
        paired_metric = paired[sample_id].get("metric_comparisons", {}).get("binary_accuracy", {})
        errors.append(
            {
                "id": sample_id,
                "error_type": error_type,
                "paired_outcome": paired_metric.get("outcome", "unresolved"),
                "reference_changeflag": reference_flag,
                "predicted_changeflag": predicted_flag,
                "binary_prediction_source": row.get("binary_prediction_source"),
                "prediction_template": _prediction_template(str(row.get("prediction", ""))),
                "reference_change_category": _change_category(str(row.get("reference", ""))),
                "reference_object_category": _object_category(str(row.get("reference", ""))),
                "prediction_object_category": _object_category(str(row.get("prediction", ""))),
                "candidate_prediction": row.get("prediction", ""),
                "reference": row.get("reference", ""),
                "baseline_prediction": baseline[sample_id].get("prediction", ""),
            }
        )

    false_negatives = [row for row in errors if row["error_type"] == "false_negative"]
    false_positives = [row for row in errors if row["error_type"] == "false_positive"]
    regressions = [row for row in errors if row["paired_outcome"] == "loss"]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis_profile": "levir_cc_text_error_attribution_v1",
        "limitations": (
            "Text-rule attribution only; categories do not replace image-level human review."
        ),
        "num_samples": len(candidate),
        "candidate_errors": {
            "total": len(errors),
            "error_rate": len(errors) / len(candidate),
            "by_type": dict(Counter(row["error_type"] for row in errors)),
        },
        "false_negative_analysis": {
            "total": len(false_negatives),
            "prediction_template": dict(
                Counter(row["prediction_template"] for row in false_negatives)
            ),
            "reference_change_category": dict(
                Counter(row["reference_change_category"] for row in false_negatives)
            ),
            "reference_object_category": dict(
                Counter(row["reference_object_category"] for row in false_negatives)
            ),
            "decision_source": dict(
                Counter(row["binary_prediction_source"] for row in false_negatives)
            ),
        },
        "false_positive_analysis": {
            "total": len(false_positives),
            "prediction_object_category": dict(
                Counter(row["prediction_object_category"] for row in false_positives)
            ),
            "decision_source": dict(
                Counter(row["binary_prediction_source"] for row in false_positives)
            ),
        },
        "paired_regressions": {
            "total": len(regressions),
            "by_error_type": dict(Counter(row["error_type"] for row in regressions)),
        },
    }
    summary["generated_figures"] = _plot(summary, output, formats)
    (output / "error_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "candidate_errors.csv", errors)
    _write_csv(output / "false_negatives.csv", false_negatives)
    _write_csv(output / "false_positives.csv", false_positives)
    _write_csv(output / "paired_regressions_attributed.csv", regressions)
    return {
        "summary": output / "error_analysis_summary.json",
        "candidate_errors": output / "candidate_errors.csv",
        "false_negatives": output / "false_negatives.csv",
        "false_positives": output / "false_positives.csv",
        "paired_regressions": output / "paired_regressions_attributed.csv",
    }


def main() -> int:
    args = parse_args()
    try:
        outputs = analyze(
            args.baseline_dir,
            args.candidate_dir,
            args.comparison_dir,
            args.output_dir,
            list(args.formats),
        )
    except ErrorAnalysisError as exc:
        print(f"Error analysis failed: {exc}")
        return 1
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
