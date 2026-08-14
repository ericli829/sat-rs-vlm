# ViT Training Infrastructure Recovery Report

Date: 2026-08-15

## Executive summary

The recovery was performed on `master` at `8b23eb6`, using the current branch as
the only canonical baseline. No reset, branch merge, commit, push, model training,
evaluation-tier regeneration, or checkpoint modification was performed.

The missing historical training capabilities are connected again:

```text
strict training config
  -> assistant-only Qwen3VLDataCollator
  -> sampler-only data selection
  -> canonical MultitaskTrainer
  -> token_mean or task_weighted loss
  -> optional current partial-ViT tuning and grouped optimizer
```

Training statistics, loss diagnostics, H1 hard-example mining, deterministic replay,
and visual-adaptation analysis are restored as reusable library modules with thin CLI
entrypoints. The current dynamic training plan, Reliability implementation, Evaluation
v1.5 pipeline, E1/E2/E3 assets, checkpoint loader, and generic visual sidecar contract
remain canonical.

## Why the code was missing

Commit `5ca9b2f58574e64fb4d542d60ffc835ba3df5661` is named `merged reliability and
vit`, but its raw commit object has only one parent, `2d8504e`. A real non-fast-forward
Git merge records two parent commits. Therefore Git never recorded
`fe7fd46696e054095c5ae1568b7f484d284d9830` as a merged parent, and later history
could not use merge ancestry to retain or reconcile all ViT-side training files.

This recovery read individual historical files with `git show`, compared them with
the current implementation, and migrated behavior module by module. The historical
training directory was not checked out over `master`.

## Recovered architecture

### Multitask loss and Trainer

- `token_mean` retains standard causal-LM behavior: one mean over all valid shifted
  assistant tokens.
- `task_weighted` first computes each sample's mean assistant-token CE and then uses:

  ```text
  loss = sum(sample_loss_i * task_weight_i) / sum(task_weight_i)
  ```

- It does not give each task group equal weight and does not regress to a global token
  mean.
- `Qwen3VLDataCollator` optionally emits `task_types`; the Trainer removes metadata and
  `labels` before model forward and invokes the configured loss strategy explicitly.
- `task_sampler.py` now contains sampler construction only. Uniform, weighted, and
  alternating-source sampling compose with one canonical `MultitaskTrainer`.
- Loss mode, task weights, diagnostics, and comparability notes are saved in the
  strategy manifest.

### Assistant-only supervision

The established mask remains unchanged:

```text
user, prompt, visual placeholder, padding -> -100
assistant answer                         -> supervised token id
```

Tokenization diagnostics reuse the same chat template and label-building function as
training. Truncation that removes all assistant targets still fails instead of silently
creating an unsupervised sample.

### Config composition

`Qwen3VLTrainingConfig` now composes all current and recovered sections:

```text
loss + statistics + hard_adaptation
+ target/max effective epochs and overtrain guard
+ vision_tuning + optimization + trainable_audit
```

Training config models reject unknown fields with `extra=forbid`. Statistics and hard
mining must use the same configured bbox area thresholds. The defaults remain:

```text
small_max  = 0.01
medium_max = 0.10
```

### Statistics and hard mining

- Statistics compute supervised-token counts from `labels != -100`, not string length.
- Reports cover source/task composition, sequence lengths, truncation, detection,
  counting, source-image properties, processor grids, and visual-token estimates when
  available.
- Fast diagnostics support deterministic per-task sampling and progress reporting.
- H1 mining consumes canonical Evaluation v1.5 evaluated predictions and retains
  task-specific scores/reasons for detection, counting, VQA, scene, caption, and change
  detection.
- Evaluation IDs are excluded, replay selection is seeded, and output manifests contain
  distributions and SHA256 checksums.
- Historical H1 `70% hard / 30% replay` remains an opt-in experimental configuration;
  it is not the default LoRA training path.
- Difficulty scoring, explicit categorization thresholds, stratified selection, and
  H1 composition are separated so H2 can reuse them. No H2 thresholds or training run
  were introduced.

### Current master capabilities preserved

- `training_plan.py` remains the source of truth for effective batch, steps per epoch,
  target effective epochs, maximum effective epochs, and overtraining protection.
- `vision_tuning.py` remains the source of truth for structural Qwen3-VL visual-module
  discovery, last-N-block unfreezing, trainable-parameter audit, and visual sidecars.
- `optimizer.py` remains the source of truth for non-overlapping LoRA, merger, and ViT
  parameter groups.
- H1 still loads an existing trainable LoRA adapter before explicit visual unfreezing.
- The current sidecar name remains `visual_trainable_weights.safetensors`; the old H1
  name is accepted only as a compatibility read path.
- Reliability source and tests were not modified.
- Evaluation v1.5 remains the only formal evaluation framework.

## Historical file disposition

| Historical ViT file or change | Current canonical implementation | Status |
|---|---|---|
| `training/losses.py` | Same path, restored strategy interface and formulas | Restored and extended with a pure-Python reference formula |
| `training/trainer.py` | Same path, one `MultitaskTrainer` | Restored and composed with samplers/current checkpoint sidecar saver |
| `training/data_statistics.py` | Same path | Restored |
| `training/hard_example_mining.py` | Same path | Restored; scoring/categorization/composition separated for future H2 |
| `training/loss_diagnostics.py` | Same path | Restored |
| `training/config.py` | Current config plus historical sections | Functionally merged; current dynamic/ViT fields retained |
| `training/optimizer.py` | Newer current implementation | Superseded historical version; not overwritten |
| `training/vision_tuning.py` | Newer current implementation | Superseded historical version; not overwritten |
| `scripts/training/analyze_training_data.py` | Same path | Restored |
| `scripts/training/build_hard_example_dataset.py` | Same path | Restored as formal thin CLI |
| `scripts/training/estimate_h1_steps.py` | Same path | Restored |
| `scripts/debug/diagnose_multitask_loss_bias.py` | Same path | Restored; read-only forward diagnostic |
| `evaluation/visual_analysis.py` and CLI | Canonical evaluated-prediction consumer | Restored as additive analysis, not a second evaluator |
| `scripts/evaluation/build_evaluation_tiers.py` | Newer current E-tier builder | Already canonical; historical version not restored |
| `scripts/evaluation/evaluate_predictions.py` | Current Evaluation v1.5 implementation | Already canonical; not overwritten |
| `scripts/evaluation/run_evaluation.py` | Current wrapper defaulting to E2 | Already canonical; not overwritten |
| H1 YAML configs | Current dynamic-plan/H1 configs | Already canonical; no fixed `max_steps=1000` restored |
| `qwen3vl_local_smoke.yaml` historical edits | Current strict-compatible config | Already canonical; only separate token-mean smoke config restored |
| Evaluation tier JSONL/manifest | Current frozen E1/E2/E3 assets | Intentionally preserved byte content/sample IDs |
| Historical evaluation tier docs | Current evaluation docs | Superseded by current E1/E2/E3 documentation |
| `test_multitask_losses.py` | Same path | Restored; torch-dependent cases skip when unavailable |
| `test_multitask_trainer.py` | Same path | Restored |
| `test_data_statistics.py` | Same path plus pure-Python companion test | Restored |
| `test_hard_example_mining.py` | Same path | Restored and extended for tier exclusion/H2 primitives |
| `test_loss_diagnostics.py` | Same path | Restored |
| `test_qwen3vl_prompt_mask.py` | Current test plus task-metadata assertion | Functionally merged |
| `test_training_config.py` | Current test plus strict/recovered-section assertions | Functionally merged |
| `test_h1_training_dry_run.py` | Same path | Restored and extended with regular LoRA dynamic-plan dry run |
| `test_h1_vision_tuning.py` | Newer current tests | Already canonical |
| `test_h1_checkpoint_loader.py` | Newer current checkpoint contract tests | Already canonical |
| `test_evaluation_tiers.py` | Newer current tests plus CRLF portability | Current canonical test retained |
| `test_v15_levir_and_comparison.py` | Current Evaluation v1.5 tests | Already canonical; historical edit not replayed |

Historical report artifacts and obsolete pre-E-tier evaluation aliases were not restored.
They are not runtime infrastructure and would conflict with the current fixed-tier model.

## Modified files

### Modified current files

- `.gitignore`
- `README.md`
- `docs/training/hard_example_visual_adaptation.md`
- `scripts/train_qwen3vl_lora.py`
- `src/sat_rs_vlm/data/qwen3vl_collator.py`
- `src/sat_rs_vlm/data/task_sampler.py`
- `src/sat_rs_vlm/evaluation/tiers.py`
- `src/sat_rs_vlm/training/config.py`
- `tests/unit/evaluation/test_evaluation_tiers.py`
- `tests/unit/test_qwen3vl_prompt_mask.py`
- `tests/unit/test_training_config.py`

### Added source and entrypoints

- `configs/train/qwen3vl_local_smoke_token_mean.yaml`
- `scripts/debug/diagnose_multitask_loss_bias.py`
- `scripts/evaluation/analyze_visual_adaptation.py`
- `scripts/training/analyze_training_data.py`
- `scripts/training/build_hard_example_dataset.py`
- `scripts/training/estimate_h1_steps.py`
- `src/sat_rs_vlm/evaluation/visual_analysis.py`
- `src/sat_rs_vlm/training/data_statistics.py`
- `src/sat_rs_vlm/training/hard_example_mining.py`
- `src/sat_rs_vlm/training/loss_diagnostics.py`
- `src/sat_rs_vlm/training/losses.py`
- `src/sat_rs_vlm/training/trainer.py`

### Added tests and reports

- `tests/integration/test_h1_training_dry_run.py`
- `tests/unit/evaluation/test_visual_analysis.py`
- `tests/unit/training/test_data_statistics.py`
- `tests/unit/training/test_data_statistics_pure.py`
- `tests/unit/training/test_hard_example_mining.py`
- `tests/unit/training/test_loss_diagnostics.py`
- `tests/unit/training/test_multitask_loss_formula.py`
- `tests/unit/training/test_multitask_losses.py`
- `tests/unit/training/test_multitask_trainer.py`
- `tests/unit/training/test_training_entry_wiring.py`
- `reports/integration/vit_training_recovery_precheck.md`
- `reports/integration/vit_training_recovery_report.md`

## E1/E2/E3 verification

No tier JSONL or manifest was regenerated. On Windows, Git checks JSONL files out with
CRLF while the manifest was produced from LF bytes. Tier validation now accepts only
the manifest hash or the exact LF-canonical JSONL hash; content changes still fail.

| Tier | Samples | Canonical SHA256 |
|---|---:|---|
| E1 | 593 | `e513ad879cfe75496b2bd4f28f076e61977861b60695a52487ad28f93c3cee07` |
| E2 | 3000 | `20ee6da734545ba213947d44bb5bb1ee930563ac1571a61c385594ecb43d7a17` |
| E3 | 62918 | `e104aefcd3a524c041479e50af95453d7c3095063bdb3ff3f075b21d2daaf48f` |

Formal evaluation defaults remain:

```text
scripts/evaluate_rs_vlm.py              -> configs/eval/qwen3vl_eval.yaml -> E2
scripts/evaluation/run_evaluation.py     -> configs/eval/qwen3vl_eval.yaml -> E2
configs/eval/evaluation_v1_5.yaml        -> tier: E2
```

E1 and E3 require explicit configs. Legacy E0/E1 experiment configs remain named,
explicit historical experiments and are not submission defaults.

## Verification results

All commands used the requested default environment:
`D:\APPS\ANACONDA\python.exe`. The incomplete project venv was not used.

```text
python -m compileall -q src scripts tests
PASS

python -m pytest tests/unit/training -q
13 passed, 5 skipped

python -m pytest tests/unit/test_training_config.py -q
5 passed

python -m pytest tests/unit/test_qwen3vl_prompt_mask.py -q
1 skipped (torch is not installed in the default local environment)

python -m pytest tests/unit -q
260 passed, 26 skipped, 1 warning

python -m pytest tests/integration/test_h1_training_dry_run.py -q
2 passed

git diff --check
PASS
```

The H1 integration fixture verifies an existing adapter path, hard-adaptation config,
partial-ViT config, and explicit max-step plan without loading a model. The regular LoRA
fixture verifies `task_weighted`, disabled vision tuning, and a dynamically resolved
one-epoch plan. CLI `--help` smoke checks passed for statistics, hard mining, loss-bias
diagnostics, H1 step estimation, and visual analysis.

An additional integration subset that does not require missing web/model dependencies
passed with `26 passed, 3 skipped`. Full integration collection is not valid in this
default environment because `torch`, `transformers`, `typer`, and `fastapi` are absent.
`ruff` is also not installed, so Ruff checks were not run. The remaining Pydantic warning
comes from the pre-existing infrastructure config's `model_id` namespace, not the strict
training config introduced here.

## Commands for a complete cloud check

```bash
python -m compileall -q src scripts tests
python -m pytest tests/unit -q
python -m pytest tests/integration -q
ruff check src tests scripts
ruff format --check src tests scripts
```

Regular LoRA dry run:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_autodl_4090.yaml \
  --dry-run
```

Historical H1 dry run:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_hard_visual_adaptation_4090.yaml \
  --dry-run
```

Default formal evaluation (E2):

```bash
python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml \
  --checkpoint /path/to/checkpoint
```

## H2 readiness and limitations

The repository now has the prerequisites for H2 Global Multitask Refinement:
configured per-sample multitask loss, exact supervision statistics, reusable difficulty
scores, explicit difficulty categorization, deterministic stratified sampling, dataset
composition primitives, strict manifests, and canonical E2 comparison.

H2 is not implemented as a final experiment in this recovery. In particular, no final
medium/core thresholds, 60/25/15 dataset, training configuration, checkpoint, prediction,
or evaluation result was created. Real-model forward/training and GPU memory behavior
must be validated in the complete AutoDL environment before starting H1 or H2.
