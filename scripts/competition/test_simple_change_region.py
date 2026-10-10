"""Test change captions followed by question injection and visual-KV constrained labels."""

from __future__ import annotations

import argparse
import copy
import html
import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sat_rs_vlm.data.prompt_templates import CHANGE_DESCRIPTION_INSTRUCTION
from replay_change_region import REGIONS, load_cases, now, write_json, write_text_atomic

LABELS = REGIONS + ("no_change",)
PRIMARY_LOCATION_INSTRUCTION = (
    "The first image is before the change and the second image is after the change. "
    "Describe the main visible change and its location in the image concisely. "
    "If there is no change, state that clearly."
)


def caption_prompt():
    # Stage one must not see the website question or any candidate labels.
    return CHANGE_DESCRIPTION_INSTRUCTION


def experiment_prompt(mode):
    if mode == "original":
        return caption_prompt()
    if mode == "primary_location":
        return PRIMARY_LOCATION_INSTRUCTION
    raise ValueError("Unknown caption mode: " + repr(mode))


def caption_sample(item_id, paths, prompt):
    return {"id": item_id, "task_type": "change_detection", "messages": [
        {"role": "user", "content": [
            *({"type": "image", "image": str(path)} for path in paths),
            {"type": "text", "text": prompt},
        ]}
    ]}


def original_sample(item_id, paths):
    return caption_sample(item_id, paths, caption_prompt())


def label_prompt(question):
    return (
        "\n\nQuestion: " + question
        + "\nReturn only one location: " + ", ".join(LABELS)
        + " (no_change means no visible change).\nAnswer: "
    )


def validate_label(answer):
    # The decoding mask, rather than text extraction or a guessed fallback,
    # must produce a complete canonical label.
    if answer not in LABELS:
        raise ValueError("Token-masked output is not a permitted label: " + repr(answer))
    return answer


def summarize(rows, total):
    def metric(selected):
        correct = sum(r["label"] == r["expected"] for r in selected)
        return {"correct": correct, "total": len(selected),
                "accuracy": correct / len(selected) if selected else None}
    scored = [r for r in rows if r["kind"] == "website" and r["expected"] in REGIONS]
    controls = [r for r in rows if r["kind"] == "identical_image_control"]
    return {
        "completed": len(rows), "total": total,
        "website_human_reference": metric(scored), "no_change_controls": metric(controls),
        "valid_labels": sum(r["label"] in LABELS for r in rows),
        "website_internal_no_change": sum(r["kind"] == "website" and r["label"] == "no_change" for r in rows),
        "original_evaluator_caption_matches": sum(
            "original_evaluator_caption" in r and r["original_evaluator_caption"] == r.get("description")
            for r in rows
        ),
        "limitations": [
            "Human region labels are not official gold; unclear annotations are excluded.",
            "Identical-image controls test no-change behavior, not natural seasonal invariance.",
            "Internal no_change cannot be sent when absent from website constraint.values.",
            "Caption correctness is available for manual review, not inferred from label accuracy.",
        ],
    }


def load_baseline(plan, cases):
    if not plan.get("baseline_results"):
        if plan.get("caption_mode", "original") != "original":
            raise ValueError("A changed prompt requires a saved original-caption baseline")
        return {}
    path = Path(plan["baseline_results"])
    metadata = json.loads((path.parent / "experiment.json").read_text(encoding="utf-8-sig"))
    if metadata["caption_prompt"] != caption_prompt():
        raise ValueError("Baseline must use the unchanged original LEVIR caption prompt")
    if metadata["label_prompt_template"] != label_prompt("{website_question}"):
        raise ValueError("Baseline label suffix differs; cannot isolate the caption prompt")
    if metadata["decoder"] != "kv_cached_prefix_constrained_greedy":
        raise ValueError("Baseline must use the same constrained decoder")
    if metadata["model"] != plan["model"]:
        raise ValueError("Baseline model configuration differs")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id = {r["item_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {c["item_id"] for c in cases}:
        raise ValueError("Baseline must contain exactly the same unique cases")
    for case in cases:
        row = by_id[case["item_id"]]
        if (row["original_request"] != case["original_request"]
                or row["kind"] != case["kind"] or row["expected"] != case["expected"]):
            raise ValueError("Baseline image order, question, contract or reference differs: " + case["item_id"])
        if row["description"] != row["original_evaluator_caption"]:
            raise ValueError("Baseline caption did not pass the original evaluator check")
    return by_id


def compare_baseline(rows, baseline):
    def metric(selected, use_baseline):
        correct = sum((baseline[r["item_id"]]["label"] if use_baseline else r["label"]) == r["expected"]
                      for r in selected)
        return {"correct": correct, "total": len(selected),
                "accuracy": correct / len(selected) if selected else None}
    scored = [r for r in rows if r["kind"] == "website" and r["expected"] in REGIONS]
    controls = [r for r in rows if r["kind"] == "identical_image_control"]
    return {
        "paired_completed": len(rows),
        "region_before": metric(scored, True), "region_after": metric(scored, False),
        "no_change_before": metric(controls, True), "no_change_after": metric(controls, False),
        "changed_descriptions": sum(r["description"] != baseline[r["item_id"]]["description"] for r in rows),
        "changed_labels": sum(r["label"] != baseline[r["item_id"]]["label"] for r in rows),
        "improved_ids": [r["item_id"] for r in scored if r["label"] == r["expected"]
                         and baseline[r["item_id"]]["label"] != r["expected"]],
        "regressed_ids": [r["item_id"] for r in scored if r["label"] != r["expected"]
                          and baseline[r["item_id"]]["label"] == r["expected"]],
    }


def render(output, cases, rows, summary, stage, experiment_id, fidelity=(), *,
           mode="original", first_stage_prompt=None, baseline=None):
    esc = lambda value: html.escape(str(value))
    baseline = baseline or {}
    first_stage_prompt = first_stage_prompt or caption_prompt()
    title = "主要变化及方位自由描述" if mode == "primary_location" else "原始自由描述"
    stage_note = (
        "第一阶段使用简短提示词，要求描述主要变化及其图中方位；不提前提供网站问题或十个候选。"
        if mode == "primary_location" else
        "第一阶段只使用原 LEVIR 提示词，完全不加入网站问题、位置要求或候选。"
    )
    by_id = {r["item_id"]: r for r in rows}
    metric = summary["website_human_reference"]
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LoRA ' + esc(title) + '与后置约束解码</title>',
        '<style>body{font:16px/1.6 system-ui;background:#eef2f6;color:#233044;margin:0}main{max-width:1100px;margin:auto;padding:24px}article,header{background:white;border:1px solid #dce3ed;border-radius:12px;padding:20px;margin:16px 0}pre{white-space:pre-wrap;background:#f5f7fa;padding:12px}figure{margin:0}img{width:390px;max-width:100%}.pair{display:flex;gap:18px;flex-wrap:wrap}.good{color:#087d52}.bad{color:#b12c3e}button,select{padding:8px;margin:5px}small{color:#65758a}</style><main>',
        '<header><h1>LoRA：' + esc(title) + ' → KV 后置注入 → 十标签约束解码</h1>',
        f'<p>状态：{esc(stage)}；完成 {len(rows)}/{len(cases)} 组。模型仅加载原 checkpoint-3406 LoRA，贪心生成，上限 128 tokens。</p>',
        f'<p>区域准确率：{metric["correct"]}/{metric["total"]}；无变化同图检查：{summary["no_change_controls"]["correct"]}/{summary["no_change_controls"]["total"]}；合法标签：{summary["valid_labels"]}/{len(rows)}。</p>',
        '<p>' + stage_note + '复用原评测 collator。保留该次生成的视觉/文本 KV 后才注入原问题及十个候选，按 token 前缀限制解码。该答题链只做一次视觉预填充，不克隆缓存。网站请求仍只有九个合法区域，内部 no_change 不直接对外发送。</p>',
        ('<p>本轮只改变第一阶段提示词。对照来自上一轮已完成的原始提示词测试，图片顺序、原问题、输出契约、模型配置和后置提示词已核对一致；本轮不要求新提示词复现历史描述。</p>'
         if baseline else
         f'<p>本机原评测生成与缓存自由描述逐字一致：{summary["original_evaluator_caption_matches"]}/{len(rows)}。另抽取的历史文件复现：{sum(r["matches_historical"] for r in fidelity)}/{len(fidelity)}；历史抽样中缓存与本机原评测一致：{sum(r["cached_matches_original"] for r in fidelity)}/{len(fidelity)}。</p>'),
        '<p>两组原标注 unclear 不计分；新增同图检查单独统计，不能替代真实无变化样本。区域评分不代表描述正确率。</p>',
        '<details><summary>实际提示词</summary><pre>' + esc(first_stage_prompt)
        + '\n\n[接在原多模态 KV 上的后缀]\n' + esc(label_prompt(cases[0]["original_request"]["question"])) + '</pre></details>',
        '<button onclick="saveReview()">导出描述人工复核</button></header>',
    ]
    if baseline:
        paired = summary["baseline_comparison"]
        before, after = paired["region_before"], paired["region_after"]
        parts.append(f'<article><h2>同题配对对照</h2><p>已完成 {paired["paired_completed"]}/{len(cases)} 组；清晰标注区域：旧提示词 {before["correct"]}/{before["total"]} → 新提示词 {after["correct"]}/{after["total"]}。描述变化 {paired["changed_descriptions"]} 组，最终标签变化 {paired["changed_labels"]} 组。</p><pre>' + esc(json.dumps(paired, ensure_ascii=False, indent=2)) + '</pre></article>')
    if fidelity:
        parts.append('<details><summary>历史预测逐条复现核对</summary>')
        for record in fidelity:
            parts.append('<p><b>' + esc(record['item_id']) + '</b></p><pre>历史文件：' + esc(record['historical_prediction'])
                         + '\n本机原评测：' + esc(record['original_evaluator_caption'])
                         + '\n缓存自由生成：' + esc(record['cached_caption']) + '</pre>')
        parts.append('</details>')
    for case in cases:
        item = case["item_id"]
        row = by_id.get(item)
        expected = case["expected"]
        css = '' if not row or expected not in LABELS else ('good' if row["label"] == expected else 'bad')
        parts += [f'<article data-item="{esc(item)}"><h2>{esc(item)}</h2><p>类型：{esc(case["kind"])}；人工/控制参考：{esc(expected)}；<b class="{css}">结果：{esc(row["label"] if row else "待运行")}</b></p>', '<div class="pair">']
        for index, image in enumerate(case["images"]):
            parts.append(f'<figure><figcaption>{"BEFORE" if index == 0 else "AFTER"}</figcaption><img src="{image["data_url"]}" alt="{esc(item)}"></figure>')
        parts.append('</div>')
        if item in baseline:
            previous = baseline[item]
            parts.append('<p>上一轮原始提示词自由描述：</p><pre>' + esc(previous["description"])
                         + '</pre><p>上一轮最终标签：<b>' + esc(previous["label"]) + '</b></p>')
        parts.append('<p>本轮模型变化描述：</p><pre>' + esc(row["description"] if row else '待运行') + '</pre>')
        if row:
            parts.append(f'<small>描述 {row["caption_seconds"]:.2f}s；原多模态 KV 约束解码 {row["label_seconds"]:.2f}s；网站输出状态：{esc(row["wire_status"])}。</small>')
        parts.append('<p>描述人工复核：<select><option value="unreviewed">未复核</option><option value="correct">正确</option><option value="partial">部分正确</option><option value="wrong">错误</option><option value="unclear">不确定</option></select></p></article>')
    parts.append('<script>function saveReview(){const review={};document.querySelectorAll("article[data-item]").forEach(a=>review[a.dataset.item]={description:a.querySelector("select").value});const payload={experiment_id:' + json.dumps(experiment_id) + ',exported_at:new Date().toISOString(),review};const u=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download=payload.experiment_id+"-description-review.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}</script></main></html>')
    write_text_atomic(output / "report.html", "\n".join(parts))


def prepare_cases(plan):
    human = json.loads(Path(plan["human_review"]).read_text(encoding="utf-8-sig"))["review"]
    cases = load_cases(Path(plan["source_capture"]), plan["source_run"],
                       Path(plan["source_score"]), Path(plan["model_package"]))
    for case in cases:
        case.update(kind="website", expected=human[case["item_id"]]["region"])
    controls = []
    for source in cases[:plan.get("no_change_controls", 5)]:
        control = copy.deepcopy(source)
        control["item_id"] = "CONTROL_IDENTICAL_" + source["item_id"]
        control["images"] = [source["images"][0], source["images"][0]]
        control["original_request"]["images"] = [source["original_request"]["images"][0]] * 2
        control["original_request"]["item_id"] = control["item_id"]
        control.update(kind="identical_image_control", expected="no_change")
        controls.append(control)
    return cases + controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if not plan["model"].get("adapter_path"):
        raise ValueError("This experiment requires the original LoRA adapter")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.jsonl").exists():
        raise ValueError("Refusing to overwrite existing experiment results")
    cases = prepare_cases(plan)
    mode = plan.get("caption_mode", "original")
    first_stage_prompt = experiment_prompt(mode)
    baseline = load_baseline(plan, cases)
    calls_per_case = 2 if baseline else 3
    planned_calls = len(cases) * calls_per_case + len(plan.get("historical_references", [])) * 2
    if plan.get("planned_calls", planned_calls) != planned_calls:
        raise ValueError("Plan model call count is inconsistent")
    experiment_id = "lora-" + mode + "-region-" + uuid.uuid4().hex
    write_json(output / "experiment.json", {
        "experiment_id": experiment_id, "prepared_at": now(), "model": plan["model"],
        "caption_prompt": first_stage_prompt, "caption_mode": mode,
        "label_prompt_template": label_prompt("{website_question}"),
        "allowed_outputs": LABELS, "do_sample": False, "max_new_tokens": 128,
        "decoder": "kv_cached_prefix_constrained_greedy", "label_input": "original_visual_and_caption_kv",
        "planned_calls": planned_calls,
        "first_stage_input": "Qwen3VLDataCollator_user_message_only",
        "baseline_results": plan.get("baseline_results"),
        "historical_predictions": plan.get("historical_predictions"),
        "website_cases": sum(case["kind"] == "website" for case in cases),
        "control_pairs": sum(case["kind"] == "identical_image_control" for case in cases),
    })
    rows = []
    fidelity = []
    engine = None
    failure_stage = "failed"
    def snapshot(stage):
        summary = summarize(rows, len(cases))
        if baseline:
            summary["baseline_comparison"] = compare_baseline(rows, baseline)
        summary["historical_fidelity"] = {
            "completed": len(fidelity), "total": len(plan.get("historical_references", [])),
            "matches_historical": sum(r["matches_historical"] for r in fidelity),
            "cached_matches_original": sum(r["cached_matches_original"] for r in fidelity),
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"stage": stage, "updated_at": now(),
                   "completed": len(rows), "total": len(cases),
                   "historical_checks_completed": len(fidelity),
                   "model_calls_completed": len(rows) * calls_per_case + len(fidelity) * 2})
        render(output, cases, rows, summary, stage, experiment_id, fidelity,
               mode=mode, first_stage_prompt=first_stage_prompt, baseline=baseline)
    snapshot("prepared")
    if args.prepare_only:
        print(json.dumps({"prepared": True, "cases": len(cases),
                          "calls": planned_calls}))
        return
    try:
        snapshot("loading_lora")
        from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
        model = plan["model"]
        engine = HuggingFaceVLMEngine(
            model_id=model["model_id"], adapter_path=model["adapter_path"],
            device=model["device"], dtype=model["dtype"], max_new_tokens=128,
            local_files_only=True, max_pixels=None, attn_implementation="sdpa",
        )
        from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
        from sat_rs_vlm.evaluation.inference import generate_predictions
        collator = Qwen3VLDataCollator(engine._processor, max_seq_length=1024,
                                      image_root=REPO, for_generation=True)
        generation = {"max_new_tokens": 128, "do_sample": False, "num_beams": 1}

        def original_generate(sample):
            return generate_predictions(engine._model, engine._processor, collator,
                                        [sample], generation, engine._torch,
                                        task_type="change_detection")[0]

        def cached_generate(sample, paths, prompt=CHANGE_DESCRIPTION_INSTRUCTION):
            return engine.reason_with_cache(prompt, paths, max_new_tokens=128,
                                            encoded_inputs=collator([sample]))

        snapshot("checking_historical_captions")
        for reference in plan.get("historical_references", []):
            if Path(plan["cancel_file"]).exists():
                raise InterruptedError("Cancellation requested")
            sample = original_sample(reference["item_id"], reference["paths"])
            plain = original_generate(sample)
            cached = cached_generate(sample, reference["paths"])
            try:
                record = {"item_id": reference["item_id"],
                          "historical_prediction": reference["prediction"],
                          "original_evaluator_caption": plain, "cached_caption": cached.reasoning_text,
                          "matches_historical": plain == reference["prediction"],
                          "cached_matches_original": plain == cached.reasoning_text}
            finally:
                cached.session.close()
            fidelity.append(record)
            write_json(output / "historical-fidelity.json", fidelity)
            snapshot("checking_historical_captions")
        if any(not r["cached_matches_original"] for r in fidelity):
            raise RuntimeError("Cached free generation differs from the unchanged original evaluator")
        if plan.get("require_historical_match", True) and any(not r["matches_historical"] for r in fidelity):
            failure_stage = "historical_reproduction_mismatch"
            raise RuntimeError("Historical captions are not exactly reproduced; region test gated off")
        snapshot("running")
        from sat_rs_vlm.competition.protocol_v2.models import AnswerContract, PredictRequest
        from sat_rs_vlm.competition.protocol_v2.answers import serialize_response
        for case in cases:
            if Path(plan["cancel_file"]).exists():
                raise InterruptedError("Cancellation requested")
            question = case["original_request"]["question"]
            paths = [im["path"] for im in case["images"]]
            sample = caption_sample(case["item_id"], paths, first_stage_prompt)
            plain_caption = (baseline[case["item_id"]]["description"] if baseline
                             else original_generate(original_sample(case["item_id"], paths)))
            started = time.perf_counter()
            reasoning = cached_generate(sample, paths, first_stage_prompt)
            description = reasoning.reasoning_text
            caption_seconds = time.perf_counter() - started
            caption_telemetry = dict(engine.last_generation_telemetry)
            try:
                if not description:
                    raise ValueError("Empty change description")
                if mode == "original" and description != plain_caption:
                    write_json(output / "caption-mismatch.json", {"item_id": case["item_id"],
                               "original_evaluator_caption": plain_caption, "cached_caption": description})
                    raise RuntimeError("Free caption differs from original evaluator; label decoding skipped")
                if Path(plan["cancel_file"]).exists():
                    raise InterruptedError("Cancellation requested")
                started = time.perf_counter()
                decoded = engine.decode_from_cache(reasoning.session, suffix=label_prompt(question), allowed_outputs=LABELS)
                label = validate_label(decoded.text)
            finally:
                reasoning.session.close()
            label_seconds = time.perf_counter() - started
            request = PredictRequest.model_validate(case["original_request"])
            compatible = label in request.response_constraint.values
            row = {
                "item_id": case["item_id"], "kind": case["kind"], "expected": case["expected"],
                "description": description, "label": label, "caption_seconds": caption_seconds,
                "original_evaluator_caption": plain_caption,
                "original_evaluator_caption_source": plan.get("baseline_results", "same_run_original_evaluator"),
                "caption_mode": mode,
                "label_seconds": label_seconds, "caption_generation": caption_telemetry,
                "label_generation": {"cache_reused": decoded.cache_reused,
                                     "latency_ms": decoded.latency_ms, "metadata": decoded.metadata,
                                     "generated_tokens": decoded.generated_tokens},
                "wire_status": "compatible" if compatible else "internal_no_change_not_in_website_values",
                "wire_response": serialize_response(request, AnswerContract.from_request(request), label) if compatible else None,
                "original_request": case["original_request"], "finished_at": now(),
            }
            if baseline:
                row["baseline_description"] = baseline[case["item_id"]]["description"]
                row["baseline_label"] = baseline[case["item_id"]]["label"]
            rows.append(row)
            with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            snapshot("running")
            print(json.dumps({"item_id": case["item_id"], "label": label, "completed": len(rows)}, ensure_ascii=False), flush=True)
        snapshot("complete")
    except BaseException as exc:
        snapshot("cancelled" if isinstance(exc, InterruptedError) else failure_stage)
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
