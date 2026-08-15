# Qwen3-VL 4B Baseline

The first 4B experiment is a base-model-only Unified E2 v2 evaluation. It does
not load a LoRA adapter and does not modify the fixed evaluation tier.

The environment variable must point to the model leaf directory containing
`config.json`, not to the parent `models/` directory:

```bash
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export DATA_ROOT=/root/autodl-tmp/datasets
```

Verify the assets before loading model weights:

```bash
test -f "$QWEN3VL_4B_MODEL_DIR/config.json"
test -f data/evaluation/tiers_v2/e2_standard.jsonl
test -f data/evaluation/tiers_v2/evaluation_tiers_manifest.json
```

Run the formal baseline:

```bash
python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_4b_baseline_e2_v2.yaml \
  --output-dir reports/evaluation/qwen3vl_4b_baseline_e2_v2
```

The output contains raw predictions plus canonical Evaluation v1.5 artifacts:

```text
reports/evaluation/qwen3vl_4b_baseline_e2_v2/
├── predictions.jsonl
├── summary.json
└── evaluation_v1_5/
    ├── evaluated_predictions.jsonl
    ├── metrics.json
    └── evaluation_manifest.json
```

If batch size 2 exceeds available memory, pass `--batch-size 1`. Changing batch
size does not change the fixed sample set or deterministic generation protocol.

Future files are prepared separately:

- `qwen3vl_4b_baseline_e1_v2.yaml` and `qwen3vl_4b_baseline_e3_v2.yaml`
- `qwen3vl_4b_lora_smoke.yaml` and `qwen3vl_4b_lora_4090.yaml`
- `qwen3vl_4b_h2_mining.yaml`
- `qwen3vl_4b_h2_global_refinement_4090.yaml`

The H2 files require `QWEN3VL_4B_REPLAY_ADAPTER_DIR`. A Qwen3-VL 2B adapter is
not a valid substitute.
