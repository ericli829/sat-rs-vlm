"""Bounded, reproducible image-header check; no pixel decoding or inference."""

import json
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image

DEST = Path(__file__).resolve().parent
ROOT = Path('F:/VIT-data/VRSBench/Images/Images_val')
report = json.loads((DEST / 'summary.json').read_text(encoding='utf-8'))
predictions = Path(report['source_directory']) / 'vrsbench/predictions.jsonl'
names = sorted({json.loads(line)['metadata']['source_image']
                for line in predictions.read_text(encoding='utf-8').splitlines() if line.strip()})
selected = random.Random(42).sample(names, min(256, len(names)))
observations, sizes, errors = [], Counter(), []
deadline = time.perf_counter() + 8
for name in selected:
    if time.perf_counter() >= deadline:
        break
    try:
        with Image.open(ROOT / name) as image:
            observations.append({'source_image': name, 'width': image.width, 'height': image.height})
            sizes[f'{image.width}x{image.height}'] += 1
    except (OSError, ValueError) as exc:
        errors.append({'source_image': name, 'error': str(exc)})
result = {
    'dataset_root': str(ROOT), 'seed': 42, 'population_unique_images': len(names),
    'requested_sample_size': len(selected), 'checked_sample_size': len(observations),
    'width_height_distribution': dict(sizes), 'observations': observations, 'errors': errors,
    'note': 'A fixed-seed sample of dataset image headers, not a full population audit. No original large-scene dimensions or historical processor grids are inferred.',
}
report['vrsbench_local_image_header_sample'] = result
(DEST / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
note = '\n## Local VRSBench Image Header Sample\n\n'
note += f'- Population: {len(names)} unique source-image filenames.\n'
note += f'- Checked: {len(observations)} files (seed 42); dimensions: {dict(sizes)}.\n'
note += '- This is a bounded sample, not verification of every image.\n'
note += f'- Errors: {errors}.\n'
with (DEST / 'summary.md').open('a', encoding='utf-8') as handle:
    handle.write(note)
print(json.dumps({k: v for k, v in result.items() if k != 'observations'}, ensure_ascii=False))
for dataset, metrics in report['datasets'].items():
    print(dataset, 'batch_latency_ms=', metrics['reconstructed_batch_latency_ms'])
    print(dataset, 'output_token_total=', metrics['retokenized_output_tokens']['total'],
          'inference_seconds_sum=', metrics['recorded_inference_seconds_sum'])
