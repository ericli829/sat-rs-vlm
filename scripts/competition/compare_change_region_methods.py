"""Compare four LEVIR-CC region-answer paths using one resident LoRA model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.competition.test_simple_change_region import (  # noqa: E402
    LABELS,
    PRIMARY_LOCATION_INSTRUCTION,
    caption_sample,
    prepare_cases,
    validate_label,
)
from replay_change_region import now, write_json, write_text_atomic  # noqa: E402

REGIONS = tuple(label for label in LABELS if label != "no_change")
METHODS = ("A", "B", "C", "D")
DOC_PATH = REPO / "docs" / "competition_change_region_integration_plan.md"
DOC_START = "<!-- REGION-COMPARISON-RESULTS:START -->"
DOC_END = "<!-- REGION-COMPARISON-RESULTS:END -->"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def encode_image(path: Path) -> str:
    content = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def make_grid_image(source: Path, output_dir: Path) -> tuple[Path, dict[str, Any], float]:
    from PIL import Image, ImageDraw

    started = time.perf_counter()
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    output = output_dir / (source_sha + "-red-grid-2px.png")
    with Image.open(source) as image:
        rendered = image.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        width, height = rendered.size
        vertical = [round(width * part / 3) for part in (1, 2)]
        horizontal = [round(height * part / 3) for part in (1, 2)]
        for x in vertical:
            draw.line((x, 0, x, height - 1), fill=(255, 0, 0), width=2)
        for y in horizontal:
            draw.line((0, y, width - 1, y), fill=(255, 0, 0), width=2)
        rendered.save(output, format="PNG")
    grid_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "source_path": str(source.resolve()),
        "source_sha256": source_sha,
        "grid_path": str(output.resolve()),
        "grid_sha256": grid_sha,
        "width": width,
        "height": height,
        "line_rgb": [255, 0, 0],
        "line_width_px": 2,
        "vertical_positions": vertical,
        "horizontal_positions": horizontal,
    }
    return output, metadata, time.perf_counter() - started


def labels_for(case: dict[str, Any]) -> tuple[str, ...]:
    return LABELS if case["kind"] == "identical_image_control" else REGIONS


def answer_suffix(question: str, labels: tuple[str, ...]) -> str:
    suffix = (
        "\n\nQuestion: " + question
        + "\nReturn only one location: " + ", ".join(labels) + "."
    )
    if "no_change" in labels:
        suffix += " (no_change means no visible change)."
    return suffix + "\nAnswer: "


def direct_prompt(question: str, labels: tuple[str, ...], *, grid: bool) -> str:
    guide = (
        "The red lines divide each image into a 3-by-3 grid. Use them only as visual guides. "
        if grid else ""
    )
    no_change = (
        " Use no_change only if the two images show no visible change."
        if "no_change" in labels else ""
    )
    return (
        "The first image is before and the second image is after. "
        + guide
        + question
        + " Choose exactly one location label from: " + ", ".join(labels) + "."
        + no_change
        + " Do not describe the images. Output only the label."
    )


def description_only_prompt(description: str, question: str, labels: tuple[str, ...]) -> str:
    no_change = (
        " Use no_change only if the description says there is no visible change."
        if "no_change" in labels else ""
    )
    return (
        "Change description: " + description
        + "\nQuestion: " + question
        + "\nUsing only the description above, choose exactly one location label from: "
        + ", ".join(labels) + "."
        + no_change
        + " Output only the label."
    )


def attach_visual_review(
    reviews: dict[str, Any], case: dict[str, Any], description: str | None
) -> dict[str, Any]:
    record = reviews.get(case["item_id"])
    hashes = [image["sha256"] for image in case["original_request"]["images"]]
    if record and description is not None and record["description"] == description and record["ordered_image_sha256"] == hashes:
        return {**record, "status": "visually_reviewed_exact_caption_and_ordered_images"}
    return {"status": "not_applicable" if description is None else "pending_exact_caption_review"}


def error_type(case: dict[str, Any], label: str, review: dict[str, Any]) -> str:
    expected = case["expected"]
    if case["kind"] == "identical_image_control":
        return "no_change_correct" if label == "no_change" else "no_change_missed"
    if expected not in REGIONS:
        return "unclear_reference_excluded"
    if label == expected:
        return "correct"
    if review.get("location") == "correct":
        return "description_direction_correct_but_label_wrong"
    if review.get("location") == "partial":
        return "description_direction_partial_and_label_wrong"
    if label == "center":
        return "center_confusion"
    return "wrong_region"


def raw_candidate_scores(diagnostics: dict[str, Any]) -> dict[str, Any]:
    scores = diagnostics.get("candidate_scores")
    if not isinstance(scores, dict) or not scores:
        raise RuntimeError("The decoder did not return complete candidate raw scores")
    return {
        label: {
            key: score[key]
            for key in (
                "token_ids", "token_log_probabilities", "label_log_probability",
                "eos_log_probability", "terminated_log_probability", "mean_token_log_probability",
            )
        }
        for label, score in scores.items()
    }


def run_decode(engine, session, suffix: str, labels: tuple[str, ...]) -> tuple[str, dict[str, Any], float]:
    started = time.perf_counter()
    decoded = engine.decode_from_cache(
        session,
        suffix=suffix,
        allowed_outputs=labels,
        record_probabilities=True,
        score_candidates=True,
    )
    elapsed = time.perf_counter() - started
    label = validate_label(decoded.text)
    diagnostics = decoded.metadata.get("probability_diagnostics", {})
    if set(diagnostics.get("candidate_scores", {})) != set(labels):
        raise RuntimeError("Candidate scoring does not cover the complete output contract")
    return label, diagnostics, elapsed


def classify_one(
    *, engine, collator, case: dict[str, Any], method: str,
    grid_paths: list[Path] | None, grid_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = labels_for(case)
    question = case["original_request"]["question"]
    paths = [Path(image["path"]) for image in case["images"]]
    description: str | None = None
    description_seconds = 0.0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    grid_seconds = 0.0
    candidate_diagnostics: dict[str, Any]
    start_total = time.perf_counter()
    session = None

    if method == "A":
        sample = caption_sample(case["item_id"], paths, PRIMARY_LOCATION_INSTRUCTION)
        encoded = collator([sample])
        started = time.perf_counter()
        reasoning = engine.reason_with_cache(
            PRIMARY_LOCATION_INSTRUCTION, paths, max_new_tokens=128, encoded_inputs=encoded
        )
        description_seconds = time.perf_counter() - started
        description = reasoning.reasoning_text
        session = reasoning.session
        try:
            label, candidate_diagnostics, decode_seconds = run_decode(
                engine, session, answer_suffix(question, labels), labels
            )
        finally:
            session.close()
    elif method == "B":
        prompt = direct_prompt(question, labels, grid=False)
        sample = caption_sample(case["item_id"], paths, prompt)
        encoded = collator([sample])
        started = time.perf_counter()
        prefill = engine.prefill_with_cache(prompt, paths, encoded_inputs=encoded)
        prefill_seconds = time.perf_counter() - started
        session = prefill.session
        try:
            label, candidate_diagnostics, decode_seconds = run_decode(
                engine, session, "\nAnswer: ", labels
            )
        finally:
            session.close()
    elif method == "C":
        sample = caption_sample(case["item_id"], paths, PRIMARY_LOCATION_INSTRUCTION)
        encoded = collator([sample])
        started = time.perf_counter()
        reasoning = engine.reason_with_cache(
            PRIMARY_LOCATION_INSTRUCTION, paths, max_new_tokens=128, encoded_inputs=encoded
        )
        description_seconds = time.perf_counter() - started
        description = reasoning.reasoning_text
        reasoning.session.close()
        prompt = description_only_prompt(description, question, labels)
        started = time.perf_counter()
        prefill = engine.prefill_with_cache(prompt, [])
        prefill_seconds = time.perf_counter() - started
        session = prefill.session
        try:
            label, candidate_diagnostics, decode_seconds = run_decode(
                engine, session, "\nAnswer: ", labels
            )
        finally:
            session.close()
    elif method == "D":
        if grid_paths is None:
            raise ValueError("Grid method requires grid image paths")
        active_paths = []
        active_metadata = []
        for source in paths:
            target, metadata, elapsed = make_grid_image(source, Path(grid_paths[0]).parent)
            active_paths.append(target)
            active_metadata.append(metadata)
            grid_seconds += elapsed
        grid_metadata.extend(active_metadata)
        prompt = direct_prompt(question, labels, grid=True)
        sample = caption_sample(case["item_id"], active_paths, prompt)
        encoded = collator([sample])
        started = time.perf_counter()
        prefill = engine.prefill_with_cache(prompt, active_paths, encoded_inputs=encoded)
        prefill_seconds = time.perf_counter() - started
        session = prefill.session
        try:
            label, candidate_diagnostics, decode_seconds = run_decode(
                engine, session, "\nAnswer: ", labels
            )
        finally:
            session.close()
    else:
        raise ValueError("Unknown comparison method: " + method)

    review = attach_visual_review(
        json.loads(Path(engine.experiment_review_path).read_text(encoding="utf-8-sig"))["review"],
        case,
        description,
    )
    scores = raw_candidate_scores(candidate_diagnostics)
    if label not in scores:
        raise RuntimeError("Selected label has no candidate score")
    best_by_sequence_likelihood = max(
        scores, key=lambda name: scores[name]["terminated_log_probability"]
    )
    return {
        "method": method,
        "item_id": case["item_id"],
        "kind": case["kind"],
        "question": question,
        "free_description": description,
        "final_label": label,
        "reference_label": case["expected"],
        "error_type": error_type(case, label, review),
        "visual_review": review,
        "raw_candidate_scores": scores,
        "best_candidate_by_full_sequence_log_likelihood": best_by_sequence_likelihood,
        "greedy_equals_full_sequence_argmax": label == best_by_sequence_likelihood,
        "decoder_diagnostics": candidate_diagnostics,
        "timing_seconds": {
            "description_generation": description_seconds,
            "prompt_prefill": prefill_seconds,
            "candidate_score_and_constrained_decode": decode_seconds,
            "grid_overlay_preprocess": grid_seconds,
            "total_end_to_end": time.perf_counter() - start_total,
        },
        "grid_images": grid_metadata[-len(paths):] if method == "D" else [],
        "original_request": case["original_request"],
        "finished_at": now(),
    }


def summarize(rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_or_none(values):
        collected = list(values)
        return statistics.mean(collected) if collected else None

    result: dict[str, Any] = {"total_rows": len(rows), "methods": {}}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        website = [
            row for row in selected
            if row["kind"] == "website" and row["reference_label"] in REGIONS
        ]
        controls = [row for row in selected if row["kind"] == "identical_image_control"]
        distribution = Counter(row["final_label"] for row in selected if row["kind"] == "website")
        control_distribution = Counter(row["final_label"] for row in controls)
        latencies = [row["timing_seconds"]["total_end_to_end"] for row in selected]
        errors = Counter(row["error_type"] for row in selected)
        correct = sum(row["final_label"] == row["reference_label"] for row in website)
        result["methods"][method] = {
            "rows": len(selected),
            "website_correct": correct,
            "website_total": len(website),
            "website_accuracy": correct / len(website) if website else None,
            "correct_ids": [row["item_id"] for row in website if row["final_label"] == row["reference_label"]],
            "no_change_correct": sum(row["final_label"] == "no_change" for row in controls),
            "no_change_total": len(controls),
            "no_change_accuracy": sum(row["final_label"] == "no_change" for row in controls) / len(controls) if controls else None,
            "website_label_distribution": dict(distribution),
            "no_change_control_label_distribution": dict(control_distribution),
            "error_counts": dict(errors),
            "direction_correct_but_label_wrong_ids": [
                row["item_id"] for row in website
                if row["error_type"] == "description_direction_correct_but_label_wrong"
            ],
            "direction_partial_but_label_wrong_ids": [
                row["item_id"] for row in website
                if row["error_type"] == "description_direction_partial_and_label_wrong"
            ],
            "mean_total_end_to_end_seconds": mean_or_none(latencies),
            "median_total_end_to_end_seconds": statistics.median(latencies) if latencies else None,
            "mean_description_seconds": mean_or_none(row["timing_seconds"]["description_generation"] for row in selected),
            "mean_prefill_seconds": mean_or_none(row["timing_seconds"]["prompt_prefill"] for row in selected),
            "mean_candidate_score_and_decode_seconds": mean_or_none(row["timing_seconds"]["candidate_score_and_constrained_decode"] for row in selected),
            "mean_grid_preprocess_seconds": mean_or_none(row["timing_seconds"]["grid_overlay_preprocess"] for row in selected),
            "greedy_differs_from_full_sequence_argmax": sum(not row["greedy_equals_full_sequence_argmax"] for row in selected),
        }
    result["paired_comparison_vs_A"] = {}
    a_by_id = {row["item_id"]: row for row in rows if row["method"] == "A"}
    for method in METHODS[1:]:
        m_by_id = {row["item_id"]: row for row in rows if row["method"] == method}
        result["paired_comparison_vs_A"][method] = {
            "label_changes": [
                {"item_id": item, "A": a_by_id[item]["final_label"], method: m_by_id[item]["final_label"],
                 "reference": a_by_id[item]["reference_label"]}
                for item in sorted(set(a_by_id) & set(m_by_id))
                if a_by_id[item]["final_label"] != m_by_id[item]["final_label"]
            ],
            "description_changes": sum(
                a_by_id[item]["free_description"] != m_by_id[item]["free_description"]
                for item in set(a_by_id) & set(m_by_id)
            ),
        }
    result["limitations"] = [
        "The 23 clear human region references are development annotations, not official gold labels; two unclear references are excluded.",
        "The five no-change controls are identical-image A/A pairs, not natural no-change scenes, and are excluded from nine-region accuracy.",
        "Candidate scores are raw model sequence log-likelihoods including EOS, not calibrated correctness probabilities.",
        "Direction correctness is counted only when the saved qualitative review matches the exact caption and ordered source-image hashes.",
    ]
    return result


def render_report(
    output: Path, cases: list[dict[str, Any]], rows: list[dict[str, Any]],
    summary: dict[str, Any], stage: str, experiment: dict[str, Any],
) -> None:
    escape = lambda value: html.escape(str(value))
    by_method_item = {(row["method"], row["item_id"]): row for row in rows}
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LEVIR-CC 九区域四方案对照</title>',
        '<style>body{font:15px/1.55 system-ui;background:#eef2f6;color:#243044;margin:0}main{max-width:1400px;margin:auto;padding:24px}header,article{background:#fff;border:1px solid #d9e1eb;border-radius:12px;padding:20px;margin:16px 0}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #dce2eb;padding:7px;vertical-align:top;text-align:left}img{width:240px;max-width:48%;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:10px}.bad{color:#b12c3e}.good{color:#087d52}small{color:#66758b}</style><main>',
        '<header><h1>LEVIR-CC 九区域定位：A/B/C/D 四方案对照</h1>',
        f'<p>状态：{escape(stage)}；完成 {len(rows)}/{len(cases)*4} 条方法记录。刷新此文件查看更新。</p>',
        f'<p>所有方法使用同一份 {escape(experiment["model"]["model_id"])} + '
        f'{escape(experiment["model"].get("adapter_path") or "无 LoRA")}、'
        f'{escape(experiment.get("dtype", "bf16"))}、{escape(experiment.get("attention", "sdpa"))}、'
        f'原 collator 与 128-token 描述上限。'
        f'{("视觉 sidecar 已加载：" + escape(experiment["visual_sidecar"]["path"]) + "。") if experiment.get("visual_sidecar") else ""}'
        'A/C 共用主要变化方位描述提示词；B/D 不生成自由描述；C 的第二阶段不输入图像。D 才叠加 2 像素纯红九宫格线。</p>',
        '<p>网站题只允许九个区域标签；5 组相同图片只在独立 no_change 控制中加入第十个标签，不计入九分类准确率。评分候选的原始对数似然包含 EOS，不能当作正确率置信度。</p>',
        '<table><tr><th>方案</th><th>清晰标注区域准确率</th><th>no_change 对照</th><th>平均端到端秒/题</th><th>标签分布</th><th>方位描述正确但标签错</th></tr>',
    ]
    for method in METHODS:
        metric = summary["methods"].get(method, {})
        distribution = metric.get("website_label_distribution", {})
        parts.append(
            f'<tr><td>{method}</td><td>{metric.get("website_correct", 0)}/{metric.get("website_total", 0)} '
            f'({(metric.get("website_accuracy") or 0):.1%})</td><td>{metric.get("no_change_correct", 0)}/{metric.get("no_change_total", 0)}</td>'
            f'<td>{(metric.get("mean_total_end_to_end_seconds") or 0):.2f}</td><td>{escape(json.dumps(distribution, ensure_ascii=False))}</td>'
            f'<td>{escape(json.dumps(metric.get("direction_correct_but_label_wrong_ids", []), ensure_ascii=False))}</td></tr>'
        )
    parts.append('</table><details><summary>完整摘要与限制</summary><pre>' + escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<details><summary>实验提示词与模型参数</summary><pre>' + escape(json.dumps(experiment, ensure_ascii=False, indent=2)) + '</pre></details></header>')
    for case in cases:
        item_id = case["item_id"]
        parts.append(f'<article><h2>{escape(item_id)} <small>{escape(case["kind"])}</small></h2><p>参考标签：<b>{escape(case["expected"])}</b>；原问题：{escape(case["original_request"]["question"])}</p><div>')
        for index, image in enumerate(case["images"]):
            parts.append(f'<figure style="display:inline-block;width:48%"><figcaption>{"BEFORE" if index == 0 else "AFTER"}</figcaption><img src="{encode_image(Path(image["path"]))}"></figure>')
        parts.append('</div><table><tr><th>方案</th><th>自由描述</th><th>最终标签 / 错误类型</th><th>端到端 / 阶段耗时</th><th>候选原始评分（log P，含 EOS）</th></tr>')
        for method in METHODS:
            row = by_method_item.get((method, item_id))
            if row is None:
                parts.append(f'<tr><td>{method}</td><td colspan="4">待运行</td></tr>')
                continue
            label_class = 'good' if row['error_type'] in {'correct', 'no_change_correct'} else 'bad'
            timings = row['timing_seconds']
            grid_preview = ""
            if method == "D" and row.get("grid_images"):
                grid_preview = "<details><summary>D 输入网格图</summary>" + "".join(
                    f'<figure style="display:inline-block;width:48%"><figcaption>{"BEFORE" if index == 0 else "AFTER"}</figcaption><img src="{encode_image(Path(image["grid_path"]))}"></figure>'
                    for index, image in enumerate(row["grid_images"])
                ) + "</details>"
            parts.append(
                f'<tr><td>{method}</td><td><pre>{escape(row["free_description"] or "（无自由描述阶段）")}</pre>{grid_preview}'
                f'<small>人工复核：{escape(row["visual_review"]["status"])}；方位判定：{escape(row["visual_review"].get("location", "N/A"))}</small></td>'
                f'<td><b class="{label_class}">{escape(row["final_label"])}</b><br>{escape(row["error_type"])}</td>'
                f'<td>{timings["total_end_to_end"]:.2f}s<br><small>描述 {timings["description_generation"]:.2f}s；预填 {timings["prompt_prefill"]:.2f}s；评分+约束解码 {timings["candidate_score_and_constrained_decode"]:.2f}s；网格 {timings["grid_overlay_preprocess"]:.4f}s</small></td>'
                '<td><details><summary>展开原始分数</summary><table><tr><th>候选</th><th>label log P</th><th>EOS log P</th><th>联合 log P</th><th>token IDs</th></tr>'
            )
            for candidate in row['raw_candidate_scores']:
                score = row['raw_candidate_scores'][candidate]
                parts.append(
                    f'<tr><td>{escape(candidate)}</td><td>{score["label_log_probability"]:.5f}</td>'
                    f'<td>{score["eos_log_probability"]:.5f}</td><td>{score["terminated_log_probability"]:.5f}</td>'
                    f'<td>{escape(score["token_ids"])}</td></tr>'
                )
            parts.append('</table></details></td></tr>')
        parts.append('</table></article>')
    parts.append('<small>逐题机器可读记录：results.jsonl；候选分数：candidate-scores.jsonl；D 网格输入清单：grid-image-manifest.json。</small></main></html>')
    write_text_atomic(output / "report.html", "\n".join(parts))


def update_document(summary: dict[str, Any], report_path: Path) -> None:
    if not DOC_PATH.exists():
        return
    text = DOC_PATH.read_text(encoding="utf-8")
    start = text.find(DOC_START)
    end = text.find(DOC_END)
    if start < 0 or end < start:
        raise RuntimeError("Documentation result markers are missing")
    method_rows = [
        "| 方案 | 清晰标注区域准确率 | no_change 控制 | 平均端到端秒/题 | 方位正确但标签错误 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for method in METHODS:
        value = summary["methods"][method]
        method_rows.append(
            f'| {method} | {value["website_correct"]}/{value["website_total"]} '
            f'({value["website_accuracy"]:.1%}) | {value["no_change_correct"]}/{value["no_change_total"]} '
            f'| {value["mean_total_end_to_end_seconds"]:.2f} | '
            f'{", ".join(value["direction_correct_but_label_wrong_ids"]) or "无已匹配人工复核项"} |'
        )
    section = (
        "四方案对照已完成，逐题输入、自由描述、候选原始对数似然、答案和阶段耗时见 `"
        + str(report_path.relative_to(REPO)).replace("\\", "/")
        + "`。\n\n"
        + "\n".join(method_rows)
        + "\n\n"
        + "本轮直接测得的结果只用于这 23 条清晰人工标注开发题；两个 unclear 不计分，5 组 A/A 无变化控制独立统计。标签评分和延迟包含完整候选评分的诊断成本。方向正确但标签错误只计入与既有逐字描述及有序图片哈希完全匹配的主观评审，因此是下界。"
        + "\n\n仍待验证的假设：文本-only 路径是否普遍弱于保留视觉 KV；九宫格是否能稳定改善坐标映射；当前小样本上方法差异是否能推广到自然无变化或边界区域样本。"
    )
    replacement = DOC_START + "\n" + section + "\n" + DOC_END
    text = text[:start] + replacement + text[end + len(DOC_END):]
    write_text_atomic(DOC_PATH, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.jsonl").exists():
        raise ValueError("Use a fresh experiment output directory")
    cases = prepare_cases(plan)
    if len(cases) != 30:
        raise ValueError(f"Expected 25 website pairs plus 5 controls; got {len(cases)}")
    grid_dir = output.parent / "grid-images"
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_manifest: dict[str, Any] = {}
    for case in cases:
        for image in case["images"]:
            source = Path(image["path"])
            target, metadata, _ = make_grid_image(source, grid_dir)
            grid_manifest[metadata["source_sha256"]] = metadata
            image["grid_path"] = str(target)
    write_json(output.parent / "grid-image-manifest.json", {
        "line_rgb": [255, 0, 0], "line_width_px": 2,
        "positions": ["1/3", "2/3"], "unique_images": grid_manifest,
    })
    reviews = json.loads(Path(plan["free_output_review"]).read_text(encoding="utf-8-sig"))["review"]
    experiment = {
        "experiment_id": "levir-region-methods-" + uuid.uuid4().hex,
        "prepared_at": now(),
        "model": plan["model"],
        "dtype": "bfloat16",
        "attention": "sdpa",
        "max_new_tokens_for_descriptions": 128,
        "max_seq_length": 1024,
        "processor": "Qwen3VLDataCollator with local model processor defaults",
        "candidate_scoring": "raw token log probabilities plus EOS; complete candidate paths share the same KV prefix",
        "description_prompt": PRIMARY_LOCATION_INSTRUCTION,
        "methods": {
            "A": "free description, then constrained nine-way decode on original image+description KV",
            "B": "image-only prompt prefill, then constrained direct nine-way decode",
            "C": "free description, then text-only prompt prefill and constrained nine-way decode",
            "D": "two images with pure-red 2px 3x3 grid overlay, then constrained direct nine-way decode",
        },
        "control_policy": "five identical-image pairs use nine region labels plus no_change; excluded from nine-region accuracy",
        "planned_rows": len(cases) * len(METHODS),
        "image_order_hashes_checked": True,
        "service_restore": "disabled by user request; this runner does not launch or restore the local server",
    }
    write_json(output / "experiment.json", experiment)
    if args.prepare_only:
        write_json(output / "summary.json", summarize([], cases))
        render_report(output, cases, [], summarize([], cases), "prepared", experiment)
        print(json.dumps({"prepared": True, "cases": len(cases), "methods": METHODS,
                          "rows": len(cases) * len(METHODS), "grid_images": len(grid_manifest)}))
        return

    status = output / "status.json"
    rows: list[dict[str, Any]] = []
    model = None
    current = "loading_model"

    def snapshot(stage: str, *, render: bool = False) -> None:
        nonlocal current
        current = stage
        summary = summarize(rows, cases)
        write_json(output / "summary.json", summary)
        write_json(status, {
            "stage": stage, "updated_at": now(), "completed": len(rows),
            "total": len(cases) * len(METHODS), "methods_completed": dict(Counter(r["method"] for r in rows)),
        })
        if render:
            render_report(output, cases, rows, summary, stage, experiment)

    snapshot(current, render=True)
    if Path(plan["cancel_file"]).exists():
        raise InterruptedError("Cancellation requested before model load")
    try:
        from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
        from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine

        model_config = plan["model"]
        model = HuggingFaceVLMEngine(
            model_id=model_config["model_id"], adapter_path=model_config["adapter_path"],
            device=model_config["device"], dtype=model_config["dtype"],
            max_new_tokens=128, local_files_only=True, max_pixels=None,
            attn_implementation="sdpa",
            max_memory=model_config.get("max_memory"),
        )
        sidecar_path = model_config.get("visual_sidecar")
        if sidecar_path:
            from sat_rs_vlm.training.vision_tuning import load_visual_sidecar

            sidecar = Path(sidecar_path)
            manifest = json.loads(
                (sidecar.parent / "visual_trainable_manifest.json").read_text(encoding="utf-8")
            )
            sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            if sidecar_sha256 != manifest.get("sha256"):
                raise ValueError("Visual sidecar SHA256 does not match its manifest")
            parameter_names = load_visual_sidecar(model._model, sidecar)
            experiment["visual_sidecar"] = {
                "path": str(sidecar.resolve()),
                "sha256": sidecar_sha256,
                "loaded_parameter_count": len(parameter_names),
                "loaded_parameter_names": parameter_names,
            }
            write_json(output / "experiment.json", experiment)
        collator = Qwen3VLDataCollator(
            model._processor, max_seq_length=1024, image_root=REPO, for_generation=True
        )
        model.experiment_review_path = str(Path(plan["free_output_review"]).resolve())
        snapshot("running", render=True)
        for case in cases:
            for method in METHODS:
                if Path(plan["cancel_file"]).exists():
                    raise InterruptedError("Cancellation requested")
                row = classify_one(
                    engine=model, collator=collator, case=case, method=method,
                    grid_paths=[Path(image["grid_path"]) for image in case["images"]]
                    if method == "D" else None,
                    grid_metadata=[],
                )
                rows.append(row)
                with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                candidate_record = {
                    "method": method, "item_id": case["item_id"],
                    "reference_label": case["expected"], "selected_label": row["final_label"],
                    "raw_candidate_scores": row["raw_candidate_scores"],
                }
                with (output / "candidate-scores.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(candidate_record, ensure_ascii=False, allow_nan=False) + "\n")
                snapshot("running")
            snapshot("running", render=True)
        summary = summarize(rows, cases)
        snapshot("complete", render=True)
        if plan.get("update_document", True):
            update_document(summary, output / "report.html")
            write_json(output / "documentation-update.json", {
                "updated": str(DOC_PATH), "updated_at": now(),
                "summary_snapshot": str(output / "summary.json"),
            })
    except BaseException as exc:
        write_json(output / "error.json", {"type": type(exc).__name__, "error": str(exc), "updated_at": now()})
        snapshot("cancelled" if isinstance(exc, InterruptedError) else "failed", render=True)
        raise
    finally:
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
