"""Read-only replay-artifact analysis; never loads weights or runs inference."""

import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from tokenizers import Tokenizer
from PIL import Image

ROOT = Path('D:/Desktop/tzb-2026/results')
SOURCE = ROOT / 'levir_eval/vrsbench_levircc_replay_formal'
ADAPTER = ROOT / 'levir_train/vrsbench_levircc_replay_formal/round_2_adapter'
TOKENIZER = ADAPTER / 'processor/tokenizer.json'
DEST = Path(__file__).resolve().parent


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def distribution(values):
    data = sorted(values)
    if not data:
        return None
    def percentile(p):
        index = (len(data) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return data[lo] + (data[hi] - data[lo]) * (index - lo)
    return dict(count=len(data), total=sum(data), mean=statistics.mean(data),
                median=statistics.median(data), p95=percentile(.95), min=data[0], max=data[-1])


def summarize(rows):
    elapsed = sum(r['inference_latency_ms'] for r in rows) / 1000
    tokens = sum(r['_output_tokens'] for r in rows)
    return {
        'samples': len(rows),
        'amortized_latency_ms': distribution([r['inference_latency_ms'] for r in rows]),
        'retokenized_output_tokens': distribution([r['_output_tokens'] for r in rows]),
        'output_characters': distribution([len(r['prediction']) for r in rows]),
        'recorded_inference_seconds_sum': elapsed,
        'sample_throughput_per_second': len(rows) / elapsed,
        'retokenized_output_throughput_per_second_NOT_decode_tps': tokens / elapsed,
    }


def main():
    started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    result = {
        'source_directory': str(SOURCE), 'adapter_directory': str(ADAPTER),
        'tokenizer': str(TOKENIZER), 'tokenizer_sha256': sha(TOKENIZER),
        'latency_semantics': 'batch collate + device transfer + generate + text decode, divided by actual batch size; CUDA synchronized',
        'latency_evidence': 'git 449bc85 evaluation/inference.py timed_predictions; replay_eval_20260805_230834.log',
        'token_semantics': 'Re-encode saved stripped prediction with checkpoint tokenizer; add_special_tokens=False. Not original generated token IDs; EOS/padding/removed text cannot be recovered.',
        'unavailable': {
            'ttft_ms': 'No first-token timestamps.',
            'planning_execution_seconds': 'No phase telemetry; these runs use direct VLM generation.',
            'pure_decode_seconds': 'No prefill/decode timing split.',
            'decode_tokens_per_second': 'Neither decode-only duration nor original output IDs was recorded.',
            'original_generation_token_count': 'Only decoded and stripped text saved, not output IDs.',
            'historical_visual_token_count': 'No image_grid_thw or recorded input visual-token counts.',
            'tiling': 'No tiling metadata; do not equate image patches with high-resolution tiles.',
            'single_request_e2e': 'Batch-amortized timings cannot measure batch=1 latency.',
            'whole_job_wall_time': 'Recorded inference excludes load/startup/scoring/report writing.',
        }, 'datasets': {},
    }
    all_rows = {}
    for name, batch_size in [('vrsbench', 16), ('levircc', 8)]:
        path = SOURCE / name / 'predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        keys = Counter(key for row in rows for key in row)
        metadata_keys = Counter(key for row in rows for key in row.get('metadata', {}))
        assert len({r['id'] for r in rows}) == len(rows), 'Duplicate IDs'
        assert all(isinstance(r['prediction'], str) and math.isfinite(r['inference_latency_ms']) and r['inference_latency_ms'] > 0 for r in rows)
        for offset in range(0, len(rows), 512):
            chunk = rows[offset:offset + 512]
            encoded = tokenizer.encode_batch([r['prediction'] for r in chunk], add_special_tokens=False)
            for row, encoding in zip(chunk, encoded):
                row['_output_tokens'] = len(encoding.ids)
        groups = defaultdict(list)
        for row in rows:
            groups[row['task_type']].append(row)
        batches = []
        for group in groups.values():
            for offset in range(0, len(group), batch_size):
                batch = group[offset:offset + batch_size]
                assert len({r['inference_latency_ms'] for r in batch}) == 1, 'Historical batch reconstruction mismatch'
                batches.append(batch[0]['inference_latency_ms'] * len(batch))
        summary = summarize(rows)
        saved_summary = json.loads((SOURCE / name / 'summary.json').read_text(encoding='utf-8'))
        assert math.isclose(summary['amortized_latency_ms']['mean'], saved_summary['overall']['inference_latency_ms'], rel_tol=1e-10)
        summary.update({
            'source_sha256': sha(path), 'configured_batch_size': batch_size,
            'field_coverage': dict(keys), 'metadata_field_coverage': dict(metadata_keys),
            'by_task': {task: summarize(group) for task, group in groups.items()},
            'reconstructed_batch_latency_ms': distribution(batches),
            'all_reconstructed_batches_have_identical_member_timings': True,
            'unique_source_images_in_metadata': len({r['metadata']['source_image'] for r in rows if r['metadata'].get('source_image')}) or None,
        })
        result['datasets'][name] = summary
        all_rows[name] = rows

    # Read local PNG headers only; a time budget avoids an unbounded disk scan.
    levir = Path('E:/迅雷下载/LEVIR-CC')
    if levir.exists():
        annotations = json.loads((levir / 'LevirCCcaptions.json').read_text(encoding='utf-8'))['images']
        lookup = {f"levircc_{entry['split']}_{entry['imgid']}_{sentence['sentid']}": entry
                  for entry in annotations for sentence in entry['sentences']}
        sizes, checked, missing, matched_ids = Counter(), set(), [], 0
        deadline = time.perf_counter() + 8
        completed = True
        for row in all_rows['levircc']:
            if time.perf_counter() > deadline:
                completed = False
                break
            entry = lookup.get(row['id'])
            if entry is None:
                missing.append(row['id'])
                continue
            matched_ids += 1
            for phase in ['A', 'B']:
                path = levir / 'images' / entry['filepath'] / phase / entry['filename']
                if path in checked:
                    continue
                checked.add(path)
                try:
                    with Image.open(path) as img:
                        sizes[f'{img.width}x{img.height}'] += 1
                except (OSError, ValueError) as exc:
                    missing.append(f'{path}: {exc}')
        result['levircc_local_image_headers'] = {
            'annotation_sha256': sha(levir / 'LevirCCcaptions.json'),
            'matched_prediction_ids': matched_ids, 'complete_scan': completed,
            'unique_files_checked': len(checked), 'width_height_distribution': dict(sizes),
            'errors': missing, 'note': 'Dataset image dimensions, not original uncropped satellite-scene dimensions or historical processor grids.',
        }
    result['analysis_runtime_seconds'] = time.perf_counter() - started
    (DEST / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 2B LoRA Replay: Offline Performance Audit', '',
             'No inference rerun, training, network access, or model weight loading.', '',
             '| Dataset/task | N | Batch | Amortized ms/sample | Output tokens/sample (re-encoded) | Samples/s | Output tokens/s (NOT decode) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, stats in result['datasets'].items():
        for label, metrics in [(name, stats), *[(task, v) for task, v in stats['by_task'].items()]]:
            lines.append(f"| {label} | {metrics['samples']} | {stats['configured_batch_size']} | {metrics['amortized_latency_ms']['mean']:.3f} | {metrics['retokenized_output_tokens']['mean']:.3f} | {metrics['sample_throughput_per_second']:.3f} | {metrics['retokenized_output_throughput_per_second_NOT_decode_tps']:.3f} |")
    lines.extend(['', '## Definitions', '', result['latency_semantics'], '', result['token_semantics'], '',
                  'Existing average_generation_length is Python string length (characters), not tokenizer tokens.', '',
                  'Batch latency was reconstructed by task and original row order, including short final batches. Every member timing agrees.',
                  'These are historical workload-specific throughputs at different batch sizes, not a fair batch=1 latency or pure decode benchmark.', '',
                  '## Unavailable From Saved Predictions', ''])
    lines.extend(f'- {key}: {reason}' for key, reason in result['unavailable'].items())
    lines.extend(['', '## Local LEVIR Image Headers', '', '```json', json.dumps(result.get('levircc_local_image_headers'), ensure_ascii=False, indent=2), '```', '',
                  '## Provenance', '', f'- Predictions: {SOURCE}', f'- Checkpoint tokenizer: {TOKENIZER}', f'- Tokenizer SHA256: {result["tokenizer_sha256"]}',
                  '- Historical timing code: git 449bc85, src/sat_rs_vlm/evaluation/inference.py',
                  '- Batch-size log: results/levir_train_logs/logs/replay_eval_20260805_230834.log', '',
                  'Detailed distributions, checksums and coverage are saved in summary.json.'])
    (DEST / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:17]))
    print(json.dumps(result.get('levircc_local_image_headers'), ensure_ascii=False))
    print('analysis_seconds=', result['analysis_runtime_seconds'])
    print('report=', DEST / 'summary.md')


if __name__ == '__main__':
    main()
