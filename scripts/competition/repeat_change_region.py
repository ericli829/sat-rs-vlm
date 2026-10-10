"""Repeat the caption-to-region experiment without loading another model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import statistics
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from replay_change_region import (
    CAPTION_PROMPT, GRID_NOTE, REGION_RULES, REGIONS, generated_request,
    load_cases, now, post, valid_region, write_json, write_text_atomic,
)


def alias_region(answer):
    """Analysis-only aliases; never extract a direction from descriptive prose."""
    if not isinstance(answer, str):
        return None
    normalized = answer.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "upper_left": "top_left", "upper_right": "top_right", "upper": "top",
        "lower_left": "bottom_left", "lower_right": "bottom_right", "lower": "bottom",
        "middle": "center", "middle_left": "left", "middle_right": "right",
        "top_center": "top", "bottom_center": "bottom", "middle_center": "center",
        "top_middle": "top", "bottom_middle": "bottom",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in REGIONS else None


def label_request(original, experiment_id, round_index, caption, *, constrained):
    question = (
        original["question"] + "\n\nEarlier model description (unverified):\n"
        + caption + "\n\n" + REGION_RULES
    )
    phase = f"r{round_index}_" + ("constrained" if constrained else "free")
    request = generated_request(original, experiment_id, phase, question)
    if constrained:
        request["response_constraint"] = copy.deepcopy(original["response_constraint"])
        request["answer_type"] = original["answer_type"]
    return request


def summarize(cases, results, human, baseline, rounds, current_baseline=None):
    eligible = {item_id: row["region"] for item_id, row in human.items() if row["region"] in REGIONS}
    summaries = []
    for index in range(1, rounds + 1):
        trial = [r for r in results if r["round"] == index]
        scored = [r for r in trial if r["item_id"] in eligible]
        row = {"round": index, "completed": len(trial), "human_scored": len(scored)}
        for method in ("free_strict", "free_alias_only", "constrained"):
            correct = sum(r[method + "_region"] == eligible[r["item_id"]] for r in scored)
            row[method] = {
                "correct": correct, "total": len(scored),
                "accuracy": correct / len(scored) if scored else None,
                "valid": sum(r[method + "_region"] is not None for r in trial),
            }
        summaries.append(row)
    stability = []
    for case in cases:
        item_id = case["item_id"]
        samples = sorted((r for r in results if r["item_id"] == item_id), key=lambda r: r["round"])
        stability.append({
            "item_id": item_id, "completed_rounds": len(samples),
            "caption_unique_count": len({r["caption"].get("answer") for r in samples}),
            "free_raw_unique_count": len({r["free"].get("answer") for r in samples}),
            "constrained_unique_count": len({r["constrained"].get("answer") for r in samples}),
            "free_labels": [r["free"].get("answer") for r in samples],
            "constrained_labels": [r["constrained_region"] for r in samples],
        })
    complete = [r for r in summaries if r["completed"] == len(cases)]
    aggregate = {}
    for method in ("free_strict", "free_alias_only", "constrained"):
        accuracies = [r[method]["accuracy"] for r in complete if r[method]["accuracy"] is not None]
        aggregate[method] = {
            "completed_rounds": len(accuracies),
            "mean_accuracy": statistics.mean(accuracies) if accuracies else None,
            "min_accuracy": min(accuracies) if accuracies else None,
            "max_accuracy": max(accuracies) if accuracies else None,
        }
    baseline_correct = sum(baseline[item_id]["baseline_region"] == region for item_id, region in eligible.items())
    current_baseline = current_baseline or {}
    current_scored = {k: v for k, v in current_baseline.items() if k in eligible}
    current_correct = sum(v['region'] == eligible[k] for k, v in current_scored.items())
    return {
        "human_eligible": len(eligible), "baseline_reused_from_original_replay": {
            "correct": baseline_correct, "total": len(eligible), "accuracy": baseline_correct / len(eligible),
        },
        "current_model_original_question": {
            "completed": len(current_baseline), "correct": current_correct,
            "total": len(current_scored),
            "accuracy": current_correct / len(current_scored) if current_scored else None,
            "answers": current_baseline,
        },
        "rounds": summaries, "aggregate": aggregate, "stability": stability,
        "limitations": [
            "Human labels are not official gold. Unclear labels are excluded, invalid outputs count as wrong.",
            "The old baseline is reused, not regenerated in this task.",
            "Free caption/label generation inherits sampling; enum cached reasoning uses do_sample=False. These are different decoding methods, not a sampling-controlled prompt-only comparison.",
            "Both final methods in each round receive the same model caption and region instructions. Human labels never enter a model request.",
            GRID_NOTE,
        ],
    }


def render(output, experiment_id, cases, results, human, baseline, summary, stage, model_variant="unspecified"):
    escape = lambda value: html.escape(str(value))
    percent = lambda value: "待完成" if value is None else f"{value:.1%}"
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>变化区域多轮对照</title>',
        '<style>body{font:16px/1.55 system-ui;background:#edf2f7;color:#233044;margin:0}main{max-width:1150px;margin:auto;padding:24px}article,header{background:white;border:1px solid #dce3ed;border-radius:12px;padding:20px;margin:16px 0}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #d8e0ea;padding:8px;text-align:left;vertical-align:top}th{background:#eff4f9}pre{white-space:pre-wrap;font:14px/1.5 ui-monospace,monospace;background:#f5f7fa;padding:12px}.ok{color:#087d52;font-weight:bold}.bad{color:#b12c3e;font-weight:bold}.pictures{display:flex;gap:18px;flex-wrap:wrap}.pictures figure{margin:0}.pictures img{width:300px;max-width:100%}.muted{font-size:14px;color:#63758b}details{margin:12px 0}</style><main>',
        '<header><h1>变化区域：多轮复测与受限选择</h1>',
        f'<p>本次模型：<b>{escape(model_variant)}</b>。</p>',
        f'<p>状态：{escape(stage)}；已完成 {len(results)}/{len(cases) * len(summary["rounds"])} 组；更新 {escape(now())}。刷新本文件查看新增结果。</p>',
        '<p>原题：Which region of the image pair contains the primary change?</p>',
        '<p>每轮重新描述变化，再将同一份描述分别用于自由生成和九标签受限选择。人工参考来自已导出的标注，未加入提示词。0145、0147 不计入准确率；非法答案按错计。</p>',
        f'<p>原路径重放（复用先前结果）：{summary["baseline_reused_from_original_replay"]["correct"]}/{summary["human_eligible"]}，{percent(summary["baseline_reused_from_original_replay"]["accuracy"])}。</p>',
        f'<p>当前模型直接回答原问题：{summary["current_model_original_question"]["correct"]}/{summary["current_model_original_question"]["total"]}，{percent(summary["current_model_original_question"]["accuracy"])}；已完成 {summary["current_model_original_question"]["completed"]}/{len(cases)} 题。</p>',
        '<p class="muted">自由描述和自由标签沿用随机采样；受限选择的内部分析关闭采样。此次衡量现有路径的稳定性和格式约束效果，不是已统一采样参数的纯提示词对比。九宫格边界及主要变化定义仍是实验约定。</p>',
        '<table><tr><th>轮次</th><th>已完成</th><th>自由生成严格准确率</th><th>只统一方向别名</th><th>受限选择准确率</th><th>合法输出（自由 / 受限）</th></tr>',
    ]
    for row in summary["rounds"]:
        cells = [row["round"], f'{row["completed"]}/{len(cases)}']
        for method in ("free_strict", "free_alias_only", "constrained"):
            metric = row[method]
            cells.append(f'{metric["correct"]}/{metric["total"]} ({percent(metric["accuracy"])})')
        cells.append(f'{row["free_strict"]["valid"]} / {row["constrained"]["valid"]}')
        parts.append('<tr>' + ''.join('<td>' + escape(c) + '</td>' for c in cells) + '</tr>')
    parts += ['</table><details><summary>查看提示词</summary><pre>' + escape(CAPTION_PROMPT + '\n\n' + REGION_RULES) + '</pre></details></header>']
    for case in cases:
        item_id = case["item_id"]
        region = human[item_id]["region"]
        parts += [f'<article><h2>{escape(item_id)}</h2><p>人工：<b>{escape(region)}</b>；原路径：{escape(baseline[item_id]["baseline_region"])}</p>', '<details><summary>查看前后原图</summary><div class="pictures">']
        for index, image in enumerate(case["images"]):
            parts.append(f'<figure><figcaption>{"BEFORE" if index == 0 else "AFTER"}</figcaption><img src="{image["data_url"]}" alt="{escape(item_id)}"></figure>')
        parts += ['</div></details><table><tr><th>轮次</th><th>变化描述</th><th>自由生成原文</th><th>受限选择</th></tr>']
        current = summary['current_model_original_question']['answers'].get(item_id)
        if current:
            parts.append(f'<tr><td>原问题</td><td>当前模型，不额外提供变化描述</td><td colspan="2">{escape(current["result"].get("answer", "未成功返回"))}</td></tr>')
        samples = sorted((r for r in results if r["item_id"] == item_id), key=lambda r: r["round"])
        for row in samples:
            def answer_cell(phase, label):
                answer = row[phase].get("answer", json.dumps(row[phase].get("response"), ensure_ascii=False))
                css = '' if region not in REGIONS else ('ok' if label == region else 'bad')
                return f'<td class="{css}">{escape(answer)}</td>'
            parts.append(f'<tr><td>{row["round"]}</td><td>{escape(row["caption"].get("answer", "未成功返回"))}</td>' + answer_cell("free", row["free_strict_region"]) + answer_cell("constrained", row["constrained_region"]) + '</tr>')
        parts.append('</table></article>')
    parts.append('</main></html>')
    write_text_atomic(output / 'report.html', '\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state", "capture", "score", "package", "human-review", "baseline-results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect-baseline", action="store_true")
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        raise ValueError("rounds must be between 1 and 10")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.jsonl').exists():
        raise ValueError("Use a fresh output directory")
    cases = load_cases(args.capture, args.source_run, args.score, args.package.resolve())
    review_bytes = args.human_review.read_bytes()
    human_export = json.loads(review_bytes.decode('utf-8-sig'))
    human = human_export['review']
    baseline = {r['item_id']: r for r in map(json.loads, args.baseline_results.read_text(encoding='utf-8').splitlines())}
    exported = {r['item_id']: r for r in human_export['cases']}
    if set(human) != {c['item_id'] for c in cases} or set(baseline) != set(human):
        raise ValueError('Human review, baseline, and captured items do not match')
    for item_id in human:
        if human[item_id]['region'] not in set(REGIONS) | {'unclear'}:
            raise ValueError('Unexpected human label')
        if exported[item_id]['baseline_region'] != baseline[item_id]['baseline_region']:
            raise ValueError('Review export does not match the baseline results')
    if not any(row['region'] in REGIONS for row in human.values()):
        raise ValueError('At least one definite human region label is required')
    state = json.loads(args.state.read_text(encoding='utf-8-sig'))
    model_config = json.loads(Path(state['config_file']).read_text(encoding='utf-8-sig'))['model']
    model_variant = 'base_only (no LoRA loaded)' if model_config.get('adapter_path') is None else 'with_lora'
    endpoint = state['endpoint'].rstrip('/')
    origin = urlsplit(endpoint)
    if origin.scheme != 'http' or origin.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('Only the local authenticated diagnostic service is supported')
    experiment_file = output / 'experiment.json'
    experiment_id = json.loads(experiment_file.read_text(encoding='utf-8'))['experiment_id'] if experiment_file.exists() else 'repeat-change-region-' + uuid.uuid4().hex
    write_json(experiment_file, {
        'experiment_id': experiment_id, 'prepared_at': now(), 'rounds': args.rounds,
        'total_item_rounds': len(cases) * args.rounds,
        'planned_model_calls': len(cases) * args.rounds * 3 + (len(cases) if args.collect_baseline else 0),
        'source_run': args.source_run, 'human_review_source': str(args.human_review.resolve()),
        'human_review_sha256': hashlib.sha256(review_bytes).hexdigest(),
        'baseline_source': str(args.baseline_results.resolve()),
        'caption_prompt': CAPTION_PROMPT, 'region_rules': REGION_RULES, 'endpoint': endpoint,
        'model_config': model_config, 'model_variant': model_variant,
    })
    (output / 'human-review.json').write_bytes(review_bytes)
    results = []
    current_baseline = {}

    def check_cancel():
        if args.cancel_file and args.cancel_file.exists():
            raise InterruptedError('Cancellation requested; restore the original service')

    def progress(stage, **extra):
        write_json(output / 'status.json', {
            'experiment_id': experiment_id, 'stage': stage, 'updated_at': now(),
            'rounds': args.rounds, 'total': len(cases) * args.rounds,
            'completed': len(results), **extra,
        })

    def artifacts(stage):
        summary = summarize(cases, results, human, baseline, args.rounds, current_baseline)
        write_json(output / 'summary.json', summary)
        render(output, experiment_id, cases, results, human, baseline, summary, stage, model_variant)

    progress('prepared')
    artifacts('prepared')
    if args.prepare_only:
        print(json.dumps({'prepared': len(cases), 'rounds': args.rounds, 'planned_model_calls': len(cases) * args.rounds * 3 + (len(cases) if args.collect_baseline else 0), 'report': str(output / 'report.html')}), flush=True)
        return
    secret = Path(state['key_file']).read_text(encoding='utf-8-sig').strip()
    if not secret:
        raise ValueError('Empty API key')
    try:
        with (output / 'responses.jsonl').open('a', encoding='utf-8', buffering=1) as raw, (output / 'results.jsonl').open('a', encoding='utf-8', buffering=1) as records:
            if args.collect_baseline:
                with (output / 'original-question-results.jsonl').open('a', encoding='utf-8', buffering=1) as originals:
                    for case in cases:
                        check_cancel()
                        original = case['original_request']
                        progress('original_question', item_id=case['item_id'])
                        request = generated_request(original, experiment_id, 'baseline', original['question'])
                        response = post(endpoint, secret, request)
                        raw.write(json.dumps({'phase': 'original_question', 'received_at': now(), 'request': request, **response}, ensure_ascii=False) + '\n')
                        raw.flush()
                        if response['status'] in {401, 503, 504}:
                            raise RuntimeError('Local service returned HTTP ' + str(response['status']))
                        item_result = {'item_id': case['item_id'], 'region': valid_region(response.get('answer'), original), 'result': response}
                        originals.write(json.dumps(item_result, ensure_ascii=False) + '\n')
                        originals.flush()
                        current_baseline[case['item_id']] = item_result
                        artifacts('collecting_original_question')
            for round_index in range(1, args.rounds + 1):
                for case in cases:
                    check_cancel()
                    original = case['original_request']
                    item_id = case['item_id']

                    def call(phase, request):
                        check_cancel()
                        progress(phase, round=round_index, item_id=item_id)
                        result = post(endpoint, secret, request)
                        raw.write(json.dumps({'round': round_index, 'phase': phase, 'received_at': now(), 'request': request, **result}, ensure_ascii=False) + '\n')
                        raw.flush()
                        if result['status'] in {401, 503, 504}:
                            raise RuntimeError('Local service returned HTTP ' + str(result['status']))
                        return result

                    caption = call('caption', generated_request(original, experiment_id, f'r{round_index}_caption', CAPTION_PROMPT))
                    free = {'status': None, 'response': {'error': 'caption_failed'}}
                    constrained = {'status': None, 'response': {'error': 'caption_failed'}}
                    if caption.get('answer'):
                        free = call('free', label_request(original, experiment_id, round_index, caption['answer'], constrained=False))
                        constrained = call('constrained', label_request(original, experiment_id, round_index, caption['answer'], constrained=True))
                    record = {
                        'round': round_index, 'item_id': item_id, 'finished_at': now(),
                        'caption': caption, 'free': free, 'constrained': constrained,
                        'free_strict_region': valid_region(free.get('answer'), original),
                        'free_alias_only_region': alias_region(free.get('answer')),
                        'constrained_region': valid_region(constrained.get('answer'), original),
                        'human_region': human[item_id]['region'],
                    }
                    records.write(json.dumps(record, ensure_ascii=False) + '\n')
                    records.flush()
                    results.append(record)
                    artifacts('running')
                    progress('running', last_round=round_index, last_item_id=item_id)
                    print(json.dumps({'round': round_index, 'completed': len(results), 'item_id': item_id, 'free': record['free_strict_region'], 'constrained': record['constrained_region']}), flush=True)
        artifacts('complete')
        progress('complete', report=str(output / 'report.html'), summary=str(output / 'summary.json'))
    except BaseException as exc:
        artifacts('failed')
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
