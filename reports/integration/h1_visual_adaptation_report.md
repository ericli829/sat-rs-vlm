# H1 Visual Adaptation Integration Report

Generated: 2026-08-11

## Modified files

H1 implementation and documentation:

- `README.md`
- `configs/train/qwen3vl_hard_visual_adaptation.yaml`
- `docs/training/hard_example_visual_adaptation.md`
- `reports/integration/h1_visual_adaptation_precheck.md`
- `reports/integration/h1_visual_adaptation_report.md`
- `scripts/train_qwen3vl_lora.py`
- `scripts/training/analyze_training_data.py`
- `scripts/training/build_hard_example_dataset.py`
- `scripts/training/estimate_h1_steps.py`
- `scripts/evaluation/analyze_visual_adaptation.py`
- `src/sat_rs_vlm/data/qwen3vl_collator.py`
- `src/sat_rs_vlm/data/task_sampler.py`
- `src/sat_rs_vlm/training/config.py`
- `src/sat_rs_vlm/training/data_statistics.py`
- `src/sat_rs_vlm/training/hard_example_mining.py`
- `src/sat_rs_vlm/training/vision_tuning.py`
- `src/sat_rs_vlm/training/optimizer.py`
- `src/sat_rs_vlm/evaluation/checkpoint_loader.py`
- `src/sat_rs_vlm/evaluation/visual_analysis.py`
- `tests/integration/test_h1_training_dry_run.py`
- `tests/unit/training/test_data_statistics.py`
- `tests/unit/training/test_hard_example_mining.py`
- `tests/unit/training/test_h1_vision_tuning.py`
- `tests/unit/evaluation/test_h1_checkpoint_loader.py`
- `tests/unit/evaluation/test_visual_analysis.py`

Existing quality-gate drift fixed without behavioral changes:

- `scripts/build_mixed_precision_quant_config.py`: wrapped one long help string.
- `scripts/evaluate_base_multidataset.py`: renamed an unused loop variable.
- `tests/unit/reliability/test_reliability_config.py`: synchronized the stale expected
  batch size with the committed `increase batch size` change (8 -> 16).
- Ruff formatting only: `scripts/evaluate_rs_vlm.py`,
  `scripts/quantization_sensitivity_test.py`, `src/sat_rs_vlm/quantization/benchmark.py`,
  `src/sat_rs_vlm/quantization/config.py`, `src/sat_rs_vlm/quantization/quantizer.py`,
  `src/sat_rs_vlm/quantization/sensitivity.py`,
  `tests/unit/compression/test_quantization.py`,
  `tests/unit/quantization/test_sensitivity.py`, and
  `tests/unit/test_multidataset_base_evaluation.py`.
- `src/sat_rs_vlm/quantization/benchmark.py` also renames a reused local variable to
  satisfy strict mypy; inference behavior is unchanged.

## New architecture

```text
Training/source data
  -> exact Collator supervision and truncation statistics
  -> train-split Evaluation v1.5 evaluated_predictions
  -> continuous hard scoring + fixed-eval-ID exclusion
  -> deterministic hard 70% + regular replay 30%
  -> existing Stage-B LoRA adapter (is_trainable=True)
  -> freeze all base parameters
  -> enable LoRA + last-N ViT blocks + main visual merger
  -> trainable parameter audit
  -> non-overlapping LoRA/merger/ViT AdamW groups
  -> PEFT adapter + visual safetensors sidecar
  -> Evaluation v1.5 paired comparison and visual analysis
```

The original assistant-only labels are unchanged. H1 remains `training.method=lora`,
uses the existing Dataset/Collator/Trainer path, and requires an initial adapter plus
an explicit `max_steps`. The formal bbox protocol remains `label+bbox` with
`normalized_0_1` coordinates.

The local Qwen3-VL checkpoint was inspected without GPU tensor loading. It contains
24 `model.visual.blocks`, one `model.visual.merger`, three
`model.visual.deepstack_merger_list` entries, and `model.visual.patch_embed`. Runtime
selection uses `len(visual.blocks)` and object structure, not those literal counts or
PEFT parameter-name prefixes.

## Commands

Set the existing environment resolver variables first:

```bash
export LOCAL_MODEL_DIR=/path/to/Qwen3-VL-2B-Instruct
export FINAL_LORA_CHECKPOINT=/path/to/stage_b/final_adapter
export DATA_ROOT=/path/to/datasets
export VAL_JSONL=/path/to/fixed_eval_or_validation.jsonl
export OUTPUT_ROOT=/path/to/outputs
export H1_MINING_TRAIN_JSONL=/path/to/training_mining_source.jsonl
export H1_MINING_EVALUATION_DIR=/path/to/train_split/evaluation_v1_5
export FIXED_EVAL_IDS=/path/to/fixed_593_evaluation_ids.jsonl
```

1. Training statistics:

```bash
python scripts/training/analyze_training_data.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml \
  --run-name h1_precheck
```

2. Hard mining and hard+replay dataset construction:

```bash
python scripts/training/build_hard_example_dataset.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml
```

3. Estimate the H1 step budget:

```bash
python scripts/training/estimate_h1_steps.py \
  --reference-stage-steps <STAGE_B_GLOBAL_STEPS>
```

4. H1 dry-run:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml \
  --dry-run
```

5. H1 AutoDL training:

```bash
source environments/autodl.env
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml
```

6. Evaluation v1.5 baseline and H1 evaluation:

```bash
python scripts/evaluation/run_evaluation.py \
  --config configs/evaluation/evaluation_v1_5.yaml \
  --checkpoint "$FINAL_LORA_CHECKPOINT"

python scripts/evaluation/run_evaluation.py \
  --config configs/evaluation/evaluation_v1_5.yaml \
  --checkpoint "$H1_CHECKPOINT"
```

7. Paired comparison and visual-factor analysis:

```bash
python scripts/evaluation/compare_evaluations.py \
  --baseline-dir reports/evaluation/stage_b \
  --candidate-dir reports/evaluation/h1 \
  --output-dir reports/evaluation/h1_comparison

python scripts/evaluation/analyze_visual_adaptation.py \
  --before reports/evaluation/stage_b \
  --after reports/evaluation/h1 \
  --output reports/evaluation/h1_comparison/visual_analysis.json
```

8. Evaluation v1.5 plotting:

```bash
python scripts/evaluation/plot_evaluation_results.py \
  --evaluation "stage_b=reports/evaluation/stage_b" \
  --evaluation "h1=reports/evaluation/h1" \
  --comparison "h1_vs_stage_b=reports/evaluation/h1_comparison" \
  --output-dir reports/evaluation/h1_comparison/figures \
  --overwrite
```

## Verification

Completed locally with the default system Python (not `.venv`):

- `python -m compileall -q src scripts`: passed.
- `ruff check src tests scripts`: passed.
- `ruff format --check src tests scripts`: passed, 244 files already formatted.
- `python -m mypy src/sat_rs_vlm`: passed, 108 source files.
- New H1 statistics/mining/vision/optimizer/visual-analysis tests: 8 passed.
- H1 dry-run/checkpoint-sidecar/reliability targeted regression: 6 passed.
- Replay coverage and evaluation-leakage tests: 2 passed.
- Original `qwen3vl_autodl_4090.yaml` LoRA dry-run: passed.
- `qwen3vl_hard_visual_adaptation.yaml` dry-run with local smoke adapter: passed.
- Real local AutoProcessor statistics smoke (one image, no model/forward): passed;
  report at `.tmp/h1-validation/statistics-smoke/summary.json` contains 12 supervised
  tokens, no truncation, `grid_thw=[1,16,16]`, and 64 approximate visual tokens.
- Fake-model partial-ViT audit verifies only last 2/4 blocks, main merger, and LoRA
  train; earlier blocks, patch embed, deep-stack mergers, language model, and
  unexpected head remain frozen.
- Full unit/integration suites: pending background execution under the repository's
  long-running-command policy. Their PID/log locations are reported to the user when
  launched and this section must be updated after completion.

No real H1 optimizer step was run locally. The existing model and local smoke adapter
were used only for asset/dry-run/processor checks; final Stage-B adapter training and
593-sample Evaluation v1.5 comparison belong on AutoDL.

## Remaining limitations

- `bbox_2d + scaled_0_1000` has not been tested or implemented.
- LoRA+ has not been tested or implemented.
- AdaLoRA has not been tested or implemented.
- Last-4 ViT blocks have not been tested; this remains H2.
- Visual resolution/token budget has not been changed; correlation is diagnostic.
- H1 final `max_steps` remains a user decision after Stage-B step and data-statistics
  review. The YAML value is a replaceable placeholder, not an automatic choice.
- Caption CIDEr-D remains the repository's single-reference approximation.
- Visual-token and image-resolution correlations are unavailable when prediction
  metadata does not contain those fields; missing values are never fabricated.
- Intermediate Trainer checkpoints contain adapter and visual sidecar, while the
  final root additionally contains the complete strategy manifest and processor used
  by the formal Evaluation v1.5 loader.
