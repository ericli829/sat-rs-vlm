# 2B LoRA Replay: Offline Performance Audit

No inference rerun, training, network access, or model weight loading.

| Dataset/task | N | Batch | Amortized ms/sample | Output tokens/sample (re-encoded) | Samples/s | Output tokens/s (NOT decode) |
|---|---:|---:|---:|---:|---:|---:|
| vrsbench | 62918 | 16 | 101.885 | 16.202 | 9.815 | 159.026 |
| captioning | 9350 | 16 | 227.018 | 53.895 | 4.405 | 237.403 |
| detection | 16153 | 16 | 122.495 | 27.070 | 8.164 | 220.985 |
| vqa | 30855 | 16 | 60.823 | 1.519 | 16.441 | 24.969 |
| counting | 6131 | 16 | 66.341 | 5.007 | 15.074 | 75.467 |
| scene_classification | 429 | 16 | 59.963 | 1.627 | 16.677 | 27.134 |
| levircc | 1333 | 8 | 100.986 | 8.904 | 9.902 | 88.170 |
| change_detection | 1333 | 8 | 100.986 | 8.904 | 9.902 | 88.170 |

## Definitions

batch collate + device transfer + generate + text decode, divided by actual batch size; CUDA synchronized

Re-encode saved stripped prediction with checkpoint tokenizer; add_special_tokens=False. Not original generated token IDs; EOS/padding/removed text cannot be recovered.

Existing average_generation_length is Python string length (characters), not tokenizer tokens.

Batch latency was reconstructed by task and original row order, including short final batches. Every member timing agrees.
These are historical workload-specific throughputs at different batch sizes, not a fair batch=1 latency or pure decode benchmark.

## Unavailable From Saved Predictions

- ttft_ms: No first-token timestamps.
- planning_execution_seconds: No phase telemetry; these runs use direct VLM generation.
- pure_decode_seconds: No prefill/decode timing split.
- decode_tokens_per_second: Neither decode-only duration nor original output IDs was recorded.
- original_generation_token_count: Only decoded and stripped text saved, not output IDs.
- historical_visual_token_count: No image_grid_thw or recorded input visual-token counts.
- tiling: No tiling metadata; do not equate image patches with high-resolution tiles.
- single_request_e2e: Batch-amortized timings cannot measure batch=1 latency.
- whole_job_wall_time: Recorded inference excludes load/startup/scoring/report writing.

## Local LEVIR Image Headers

```json
{
  "annotation_sha256": "8c8c3f494867b796306b98713e16ff72cbae1cf4a0a0c36931da16af9602daed",
  "matched_prediction_ids": 330,
  "complete_scan": false,
  "unique_files_checked": 660,
  "width_height_distribution": {
    "256x256": 660
  },
  "errors": [],
  "note": "Dataset image dimensions, not original uncropped satellite-scene dimensions or historical processor grids."
}
```

## Provenance

- Predictions: D:\Desktop\tzb-2026\results\levir_eval\vrsbench_levircc_replay_formal
- Checkpoint tokenizer: D:\Desktop\tzb-2026\results\levir_train\vrsbench_levircc_replay_formal\round_2_adapter\processor\tokenizer.json
- Tokenizer SHA256: be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506
- Historical timing code: git 449bc85, src/sat_rs_vlm/evaluation/inference.py
- Batch-size log: results/levir_train_logs/logs/replay_eval_20260805_230834.log

Detailed distributions, checksums and coverage are saved in summary.json.

## Local VRSBench Image Header Sample

- Population: 9350 unique source-image filenames.
- Checked: 256 files (seed 42); dimensions: {'512x512': 256}.
- This is a bounded sample, not verification of every image.
- Errors: [].

## Conditional Visual-Token Estimate (Not Historical Telemetry)

The saved Round-2 processor configuration specifies patch_size=16 and merge_size=2.
If the sampled images reach the vision encoder at their inspected dimensions, the
merged spatial-token count is (height / 16) * (width / 16) / 2^2 for one still image:

- VRSBench 512x512: 256 image tokens per image.
- LEVIR-CC 256x256: 64 per image, 128 for an A/B image pair.

These exclude image boundary markers and text tokens. They are conditional estimates,
not counts recovered from the historical run: its predictions have no image_grid_thw,
and upstream resize/truncation behavior was not recorded per sample. No 42-tile
high-resolution inference pipeline appears in the inspected historical evaluator.
