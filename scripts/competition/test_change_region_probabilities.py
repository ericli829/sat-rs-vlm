"""Repeat the same LoRA experiment while recording token and complete-label probabilities."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.competition.test_simple_change_region import (
    LABELS, caption_sample, experiment_prompt, label_prompt, load_baseline,
    prepare_cases, summarize, validate_label,
)
from replay_change_region import now, write_json, write_text_atomic


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def attach_review(review, case, description):
    record = review.get(case["item_id"])
    hashes = [image["sha256"] for image in case["original_request"]["images"]]
    if record and record["description"] == description and record["ordered_image_sha256"] == hashes:
        return {**record, "status": "visually_reviewed_same_images_and_exact_caption"}
    return {"status": "pending_visual_review", "reason": "No exact caption and ordered-image match to the saved visual review"}


def aggregate(rows, cases, rounds):
    result = summarize(rows, len(cases) * rounds)
    result["rounds"] = [
        {"round": n, **summarize([r for r in rows if r["round"] == n], len(cases))}
        for n in range(1, rounds + 1)
    ]
    result["label_counts"] = dict(Counter(r["label"] for r in rows))
    result["exact_caption_matches_previous_same_prompt"] = sum(r["same_prompt_caption_match"] for r in rows)
    result["exact_label_matches_previous_same_prompt"] = sum(r["same_prompt_label_match"] for r in rows)
    result["visual_review_coverage"] = sum(r["free_output_review"]["status"].startswith("visually_reviewed") for r in rows)
    result["complete_candidate_scores_recorded"] = sum(len(r["probability_diagnostics"].get("candidate_scores", {})) == len(LABELS) for r in rows)
    result["greedy_disagrees_with_complete_label_argmax"] = sum(
        r["label"] != max(r["probability_diagnostics"]["candidate_scores"],
                          key=lambda label: r["probability_diagnostics"]["candidate_scores"][label]["normalized_terminated_probability"])
        for r in rows
    )
    result["stability"] = [{
        "item_id": case["item_id"],
        "completed_rounds": len(selected := [r for r in rows if r["item_id"] == case["item_id"]]),
        "distinct_captions": len({r["description"] for r in selected}),
        "distinct_labels": len({r["label"] for r in selected}),
    } for case in cases]
    return result


def render(output, cases, rows, summary, stage, prompt, experiment_id):
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><meta charset="utf-8"><title>变化区域概率诊断与自由描述复核</title>',
        '<style>body{font:16px/1.6 system-ui;color:#243044;background:#edf1f6}main{max-width:1200px;margin:auto}article,header{background:white;padding:20px;margin:18px 0;border-radius:12px}img{width:384px;max-width:48%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dce2eb;padding:6px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:10px}small{color:#65758a}progress{width:90px}summary{cursor:pointer}</style><main>',
        '<header><h1>3 轮重放：变化区域概率诊断与自由描述复核</h1>',
        f'<p>状态：{esc(stage)}；完成 {len(rows)}/{summary["total"]} 组。报告每轮更新，status.json 每组更新。</p>',
        '<p>同一 LoRA、同一提示词、同一图像顺序，保持贪心解码。原视觉 KV 原位复用；完整候选评分顺序回退到后缀前缀，不复制十份缓存。诊断评分不改变实际贪心选择规则。</p>',
        '<p>每步归一化概率只针对合法的不同 token，共享前缀的标签共享一个分支。十标签概率另由完整标签各 token 与 EOS 的联合对数概率归一化；同时保存不含 EOS 及长度平均的得分供比较。这些数值不是答案正确率。</p>',
        '<p>自由描述复核来自逐对查看原图，内容与方位分别判断。仅对有序双图 SHA256 和描述逐字匹配的结果沿用，其他结果标记待复核；该判断不是官方标准答案。</p>',
        '<details><summary>实际提示词</summary><pre>' + esc(prompt) + '\n\n[自由生成后注入]\n' + esc(label_prompt(cases[0]["original_request"]["question"])) + '</pre></details>',
        '<table><tr><th>轮次</th><th>区域（人工参考）</th><th>同图无变化最终标签</th><th>合法输出</th></tr>']
    for item in summary["rounds"]:
        region, control = item["website_human_reference"], item["no_change_controls"]
        parts.append(f'<tr><td>{item["round"]}</td><td>{region["correct"]}/{region["total"]}</td><td>{control["correct"]}/{control["total"]}</td><td>{item["valid_labels"]}/{item["completed"]}</td></tr>')
    parts.append('</table><p>三轮为重复测量同一批题目，不是新增独立测试集。</p></header>')
    for case in cases:
        item_id = case["item_id"]
        parts.append('<article><h2>' + esc(item_id) + '</h2><p>类型：' + esc(case["kind"]) + '；人工/控制参考：' + esc(case["expected"]) + '</p>')
        for image in case["images"]:
            parts.append('<img src="' + image["data_url"] + '" alt="' + esc(item_id) + '">')
        for row in [r for r in rows if r["item_id"] == item_id]:
            parts += [f'<h3>第 {row["round"]} 轮：{esc(row["label"])}</h3><pre>' + esc(row["description"]) + '</pre>',
                      '<p>自由描述复核：' + esc(json.dumps({k:v for k,v in row["free_output_review"].items()
                                                       if k in {"status", "content", "location", "note", "reason"}}, ensure_ascii=False)) + '</p>',
                      '<details><summary>十个完整候选的归一化概率</summary><table><tr><th>标签</th><th>含 EOS 联合概率归一化</th><th>标签概率归一化</th><th>长度平均得分归一化</th><th>含 EOS 对数概率</th><th>token 数</th></tr>']
            scores = row["probability_diagnostics"]["candidate_scores"]
            for label, score in sorted(scores.items(), key=lambda item: -item[1]["normalized_terminated_probability"]):
                p = score["normalized_terminated_probability"]
                parts.append(f'<tr><td>{esc(label)}</td><td><progress max="1" value="{p}"></progress> {p:.4%}</td><td>{score["normalized_label_probability"]:.4%}</td><td>{score["normalized_mean_token_score"]:.4%}</td><td>{score["terminated_log_probability"]:.3f}</td><td>{len(score["token_ids"])}</td></tr>')
            parts.append('</table></details><details><summary>实际贪心步骤：约束前后 token 概率、原始 top 10、分词边界</summary><pre>' + esc(json.dumps(row["probability_diagnostics"], ensure_ascii=False, indent=2)) + '</pre></details>')
            first = row["probability_diagnostics"]["steps"][0]
            best = max(scores, key=lambda label:scores[label]["normalized_terminated_probability"])
            top = first["unconstrained_top_tokens"][0]
            parts.append(f'<p>首步合法 token 的原始总概率：{first["permitted_probability_mass"]:.3e}；对数总概率：{first["permitted_log_probability_mass"]:.3f}；未约束最高 token：<code>{esc(repr(top["text"]))}</code>；完整候选得分最高：<code>{esc(best)}</code>。</p>')
        parts.append('</article>')
    parts.append('<small>实验：' + esc(experiment_id) + '；原始逐条记录见 results.jsonl，概率专表见 probabilities.jsonl。</small></main>')
    write_text_atomic(output / "report.html", '\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    cases = prepare_cases(plan)
    baseline = load_baseline(plan, cases)
    if not baseline:
        raise ValueError('Probability repeats require the verified saved baseline')
    prompt = experiment_prompt(plan['caption_mode'])
    previous_path = Path(plan['previous_same_prompt_results'])
    previous = {r['item_id']:r for r in read_rows(previous_path)}
    previous_meta = json.loads((previous_path.parent/'experiment.json').read_text(encoding='utf-8-sig'))
    if previous_meta['caption_prompt'] != prompt or previous_meta['model'] != plan['model']:
        raise ValueError('The previous comparison must use the same prompt and model')
    for case in cases:
        if previous[case['item_id']]['original_request'] != case['original_request']:
            raise ValueError('Previous same-prompt image order or question differs')
    reviews = json.loads(Path(plan['free_output_review']).read_text(encoding='utf-8-sig'))['review']
    rounds = int(plan['repeat_count'])
    if rounds < 1 or plan['planned_calls'] != len(cases)*rounds*2:
        raise ValueError('Invalid repeat count or planned call count')
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    if (output/'results.jsonl').exists():
        raise ValueError('Refusing to overwrite an existing probability run')
    experiment_id = 'change-region-probability-' + uuid.uuid4().hex
    write_json(output/'experiment.json', {
        'experiment_id':experiment_id, 'prepared_at':now(), 'model':plan['model'],
        'caption_prompt':prompt, 'label_prompt_template':label_prompt('{website_question}'),
        'allowed_outputs':LABELS, 'repeat_count':rounds, 'do_sample':False,
        'max_new_tokens':128, 'record_probabilities':True, 'score_candidates':True,
        'candidate_cache_mode':'crop_to_suffix_prefix_between_candidates',
        'baseline_results':plan['baseline_results'], 'previous_same_prompt_results':str(previous_path),
        'free_output_review':plan['free_output_review'], 'planned_calls':plan['planned_calls'],
        'probability_warning':'Normalized text likelihood is not probability of answer correctness',
    })
    rows=[]; engine=None; calls=0
    def snapshot(stage, render_report=False):
        summary=aggregate(rows,cases,rounds)
        write_json(output/'summary.json',summary)
        write_json(output/'status.json',{'stage':stage,'updated_at':now(),'completed':len(rows),
                   'total':len(cases)*rounds,'model_calls_completed':calls,'planned_calls':plan['planned_calls'],
                   'rounds':rounds,'complete_rounds':len(rows)//len(cases)})
        if render_report:
            render(output,cases,rows,summary,stage,prompt,experiment_id)
    snapshot('prepared',True)
    if args.prepare_only:
        print(json.dumps({'prepared':True,'pairs':len(cases),'rounds':rounds,'calls':plan['planned_calls']}));return
    try:
        from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
        from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
        from sat_rs_vlm.competition.protocol_v2.models import AnswerContract, PredictRequest
        from sat_rs_vlm.competition.protocol_v2.answers import serialize_response
        snapshot('loading_lora')
        model=plan['model']
        engine=HuggingFaceVLMEngine(model_id=model['model_id'],adapter_path=model['adapter_path'],
            device=model['device'],dtype=model['dtype'],max_new_tokens=128,
            local_files_only=True,max_pixels=None,attn_implementation='sdpa')
        collator=Qwen3VLDataCollator(engine._processor,max_seq_length=1024,image_root=REPO,for_generation=True)
        for round_number in range(1,rounds+1):
            for case in cases:
                if Path(plan['cancel_file']).exists():raise InterruptedError('Cancellation requested')
                paths=[im['path'] for im in case['images']]
                sample=caption_sample(case['item_id'],paths,prompt)
                started=time.perf_counter()
                reasoning=engine.reason_with_cache(prompt,paths,max_new_tokens=128,encoded_inputs=collator([sample]))
                calls+=1
                caption_seconds=time.perf_counter()-started
                caption_telemetry=dict(engine.last_generation_telemetry)
                description=reasoning.reasoning_text
                try:
                    if not description:raise ValueError('Empty free caption')
                    if Path(plan['cancel_file']).exists():raise InterruptedError('Cancellation requested')
                    started=time.perf_counter()
                    decoded=engine.decode_from_cache(reasoning.session,suffix=label_prompt(case['original_request']['question']),
                        allowed_outputs=LABELS,record_probabilities=True,score_candidates=True)
                    calls+=1
                finally:
                    reasoning.session.close()
                label_seconds=time.perf_counter()-started
                label=validate_label(decoded.text)
                diagnostic=decoded.metadata['probability_diagnostics']
                if len(diagnostic.get('candidate_scores',{}))!=len(LABELS):
                    raise RuntimeError('Complete ten-label probabilities are missing')
                request=PredictRequest.model_validate(case['original_request'])
                compatible=label in request.response_constraint.values
                old=previous[case['item_id']]
                row={'round':round_number,'item_id':case['item_id'],'kind':case['kind'],'expected':case['expected'],
                     'description':description,'label':label,'caption_seconds':caption_seconds,'label_seconds':label_seconds,
                     'caption_generation':caption_telemetry,
                     'label_generation':{'cache_reused':decoded.cache_reused,'latency_ms':decoded.latency_ms,
                                         'metadata':{k:v for k,v in decoded.metadata.items() if k!='probability_diagnostics'}},
                     'probability_diagnostics':diagnostic,
                     'original_evaluator_caption':baseline[case['item_id']]['description'],
                     'same_prompt_caption_match':description==old['description'],'same_prompt_label_match':label==old['label'],
                     'free_output_review':attach_review(reviews,case,description),
                     'wire_response':serialize_response(request,AnswerContract.from_request(request),label) if compatible else None,
                     'original_request':case['original_request'],'finished_at':now()}
                rows.append(row)
                with (output/'results.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                with (output/'probabilities.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'round':round_number,'item_id':case['item_id'],
                                             'label':label,**diagnostic},ensure_ascii=False,allow_nan=False)+'\n')
                snapshot('running')
                print(json.dumps({'round':round_number,'item_id':case['item_id'],'label':label,'completed':len(rows)}),flush=True)
            snapshot('running',True)
        snapshot('complete',True)
    except BaseException as exc:
        write_json(output/'error.json',{'type':type(exc).__name__,'error':str(exc),'updated_at':now()})
        snapshot('cancelled' if isinstance(exc,InterruptedError) else 'failed',True)
        raise
    finally:
        if engine is not None:engine.close()


if __name__=='__main__':
    main()
