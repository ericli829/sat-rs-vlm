#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, re, sys, time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import transformers
from peft import PeftModel

REPO = Path('/root/autodl-tmp/sat-rs-vlm')
sys.path.insert(0, str(REPO / 'src'))
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class
from sat_rs_vlm.training.utils import model_input_device, move_to_device

LABELS = {'top_left','top','top_right','left','center','right','bottom_left','bottom','bottom_right'}

def generate(model: Any, processor: Any, collator: Any, samples: list[dict[str, Any]], max_new_tokens: int) -> list[str]:
    batch = collator(samples)
    prompt_width = int(batch['input_ids'].shape[-1])
    batch = move_to_device(batch, model_input_device(model, torch), torch)
    with torch.inference_mode():
        ids = model.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                             use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
    return [str(s).strip() for s in processor.batch_decode(ids[:, prompt_width:], skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)]

def with_oom_fallback(model: Any, processor: Any, collator: Any, samples: list[dict[str, Any]], max_new_tokens: int) -> list[str]:
    try:
        return generate(model, processor, collator, samples, max_new_tokens)
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower() or len(samples) <= 1:
            raise
        print(f'CUDA OOM batch={len(samples)}; retrying smaller batches', flush=True)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        mid = len(samples) // 2
        return (with_oom_fallback(model, processor, collator, samples[:mid], max_new_tokens) +
                with_oom_fallback(model, processor, collator, samples[mid:], max_new_tokens))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-seq-length', type=int, default=1024)
    ap.add_argument('--max-new-tokens', type=int, default=96)
    args = ap.parse_args()
    rows = list(Qwen3VLDataset(args.input))
    if len(rows) != 25: raise ValueError(f'expected 25 samples, got {len(rows)}')
    print(f'loaded {len(rows)} requests; loading processor/model', flush=True)
    load_started = time.perf_counter()
    processor = transformers.AutoProcessor.from_pretrained(str(args.base_model), trust_remote_code=True, local_files_only=True)
    if getattr(processor, 'tokenizer', None) is not None: processor.tokenizer.padding_side = 'left'
    model = compatible_model_class(transformers).from_pretrained(
        str(args.base_model), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='auto')
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model.eval()
    collator = Qwen3VLDataCollator(processor, args.max_seq_length, args.image_root, for_generation=True)
    load_seconds = time.perf_counter() - load_started
    print(f'model ready in {load_seconds:.1f}s; device={model.device}; batch={args.batch_size}', flush=True)
    records = []
    started = time.perf_counter()
    for begin in range(0, len(rows), args.batch_size):
        batch = rows[begin:begin + args.batch_size]
        predictions = with_oom_fallback(model, processor, collator, batch, args.max_new_tokens)
        for row, pred in zip(batch, predictions, strict=True):
            raw = str(pred).strip()
            normalized = raw.lower().strip().strip(' .,:;"\'`')
            records.append({'id': row['id'], 'prediction': raw,
                            'format_valid': normalized in LABELS,
                            'normalized_if_valid': normalized if normalized in LABELS else None,
                            'question': next((c.get('text','') for m in row['messages'] for c in (m.get('content') or []) if c.get('type') == 'text'), ''),
                            'metadata': row.get('metadata', {})})
        print(f'done={len(records)}/25 elapsed={time.perf_counter()-started:.1f}s', flush=True)
    elapsed = time.perf_counter() - started
    summary = {'route':'route_b_stage2_batch8','adapter':str(args.adapter),'base_model':str(args.base_model),
               'input':str(args.input),'samples':len(records),'batch_size':args.batch_size,
               'load_seconds':load_seconds,'inference_seconds':elapsed,
               'valid_single_region_count':sum(r['format_valid'] for r in records),
               'valid_single_region_rate':sum(r['format_valid'] for r in records)/len(records),
               'prediction_distribution':dict(Counter(r['normalized_if_valid'] for r in records if r['format_valid'])),
               'gold_labels_available':False,'accuracy_computed':False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'predictions':records},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0

if __name__ == '__main__': raise SystemExit(main())