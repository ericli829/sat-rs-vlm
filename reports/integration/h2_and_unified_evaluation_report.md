# H2 and Unified Evaluation Implementation Report

Date: 2026-08-15

## Status

Implementation, strict configuration, synthetic protocol tests, and real annotation
audits are complete. No model training, model inference, checkpoint writes, commit, or
push was performed.

The final Unified v2 JSONL freeze is intentionally pending because the current process
has no `F:` drive and therefore cannot access the VRSBench image root. The production
builder requires every image to exist under common `DATA_ROOT` and fails closed. This
report does not claim that unavailable VRSBench images passed physical validation.

`configs/eval/qwen3vl_eval.yaml` and the AutoDL submission config have been migrated to
Unified E2 v2, so the v2 population/tier generation commands below are a required data
preparation step before the next formal evaluation.

## Existing Legacy Tiers

The old configuration has exactly one evaluation source:
`data/processed/qwen3vl_val.jsonl`. Its 62918 unique rows all have
`metadata.dataset=VRSBench`; the existing E1/E2/E3 are therefore VRSBench-only.

They are now explicitly named `legacy-vrs-v1` and remain under
`data/evaluation/tiers/`. No legacy JSONL or manifest was modified or regenerated.

| Tier | Samples | Canonical LF SHA256 | Status |
|---|---:|---|---|
| E1 | 593 | `e513ad879cfe75496b2bd4f28f076e61977861b60695a52487ad28f93c3cee07` | unchanged |
| E2 | 3000 | `20ee6da734545ba213947d44bb5bb1ee930563ac1571a61c385594ecb43d7a17` | unchanged |
| E3 | 62918 | `e104aefcd3a524c041479e50af95453d7c3095063bdb3ff3f075b21d2daaf48f` | unchanged |

Historical Stage A/B, Replay, H1, quantization, Reliability, predictions, and paired
comparisons continue to use these exact identities.

## Unified Legal Population

### VRSBench

- Validation JSONL rows: 62918
- Unique IDs: 62918
- Unique source images: 9350
- Evaluation unit: one annotation/task case per JSONL row
- Train rows: 142390; unique train IDs: 142377; no train/validation ID overlap. The
  existing multi-source quota preparation still enforces unique IDs on its selected
  output, so H2 builders consume the validated prepared file rather than raw VRS rows.

### LEVIR-CC

The local annotation evidence is:

- Raw validation caption rows: 6665
- Unique validation IDs: 6665
- Unique validation image pairs: 1333
- Exactly five references per image pair
- `changeflag=0`: 3330 caption rows
- `changeflag=1`: 3335 caption rows
- Raw train caption rows: 34075, with no train/validation ID overlap

The existing canonical preparation sets `validation_group_by_images=true`. Real local
execution of the new full-population mode validated all LEVIR image paths and produced:

```text
validation_rows_available = 6665
unique_image_groups       = 1333
evaluation_cases_selected = 1333
evaluation_unit           = image_pair
changeflag=0 cases         = 666
changeflag=1 cases         = 667
```

Thus Unified E3 is 62918 VRSBench cases plus 1333 LEVIR image-pair cases, for 64251
legal cases. It is not a 1024/1333 truncated validation sample and does not treat five
captions for one pair as five independent cases.

## Unified Tier v2 Allocation

The production selection was run in read-only planning mode over the complete audited
rows. Final asset SHA256 values are not reported until the physical VRSBench path check
and freeze command run successfully.

| Tier | Total | VRSBench | LEVIR-CC |
|---|---:|---:|---:|
| E1 | 593 | 474 | 119 |
| E2 | 3000 | 2400 | 600 |
| E3 | 64251 | 62918 | 1333 |

Task distributions:

| Task | E1 | E2 | E3 |
|---|---:|---:|---:|
| captioning | 92 | 440 | 9350 |
| detection | 106 | 579 | 16159 |
| counting | 84 | 364 | 6374 |
| scene_classification | 71 | 257 | 3197 |
| vqa | 121 | 760 | 27838 |
| change_detection | 119 | 600 | 1333 |

LEVIR subtype distributions:

| Tier | changeflag=0 | changeflag=1 |
|---|---:|---:|
| E1 | 60 | 59 |
| E2 | 300 | 300 |
| E3 | 666 | 667 |

E2 VRSBench task targets are calculated from `p(task) proportional to sqrt(N_task)`.
Detection E2 contains 193 small, 193 medium, and 193 large cases. Counting and VQA
subtypes use equal diagnostic allocation with explicit capacity redistribution; no
subtype is inferred from prompt text.

E2 dataset targets are dynamically calculated as `3000 * {VRSBench: 0.8,
LEVIR-CC: 0.2}`. E1 uses the same weighting. E1 is selected from E2, and E2 from E3,
so subset invariants are structural rather than post-hoc assertions. Synthetic tests
verify unique IDs, deterministic JSONL hashes, both datasets, all six tasks, both LEVIR
flags, manifest/JSONL distributions, path validation, leakage rejection, and subset
relations.

## Portable Image Roots

`prepare_multisource_training_data.py --full-evaluation-only` reuses the established
stale-path repair and portable path rewrite. Unified rows use paths such as:

```text
VRSBench/Images/Images_val/...
LEVIR-CC/images/val/A/...
LEVIR-CC/images/val/B/...
```

The v2 tier builder checks every path under common `${DATA_ROOT}`. Failure includes
sample ID, dataset, original path, and resolved path. There is no skip option in the
formal v2 protocol.

## Paired Comparison Identity

Evaluation manifests now record:

```text
evaluation_tier
evaluation_tier_version
evaluation_tier_sha256
```

Missing version metadata on an old result is interpreted as `legacy-vrs-v1`. Comparison
rejects different versions before hash comparison, so legacy E2 and Unified E2 cannot be
paired despite sharing the label `E2`.

## H2 Mining Candidates

The candidate builder consumes the prepared full multisource training JSONL and Unified
E3 v2 manifest. It does not load a model.

Default policy:

```text
target_samples = 6000
VRSBench       = 75%
LEVIR-CC       = 25%
VRS tasks      = sqrt(population) softened allocation
subtypes       = deterministic diagnostic allocation
seed           = 42
```

It outputs:

```text
data/processed/h2/h2_mining_candidates.jsonl
data/processed/h2/h2_mining_candidates_manifest.json
```

The manifest records config snapshot, Replay checkpoint identifier, source input SHA,
protected E3 version/SHA, requested and actual allocations, distributions, all candidate
IDs, duplicate/leakage checks, and output SHA.

Inference is a separate step:

```text
h2_mining_candidates.jsonl
  -> AutoDL Replay generalist inference
  -> Evaluation v1.5
  -> evaluated_predictions.jsonl
```

The formal H2 mining workflow is cloud-side. Candidate images, the Replay
adapter, inference predictions, Evaluation v1.5 outputs, and final H2 data stay
on AutoDL. The local workstation is used for code validation and synthetic
tests only.

## H2 Final Dataset

The final builder reuses `score_training_samples()` from the restored hard-example
module. It does not recalculate Evaluation metrics.

Default final composition:

```text
target_samples          = 8000
regular_representative  = 60%
medium_hard             = 25%
core_hard               = 15%
source weights          = VRSBench 75% / LEVIR-CC 25%
```

Difficulty and source margins are independent. For each source/task cell, evaluated
candidates are stably sorted by `hard_score DESC, id ASC`; top cell quota becomes core,
the next quota becomes medium. Insufficient evaluated candidates in any cell is a hard
error and cannot borrow from another task. This avoids discrete VQA scores monopolizing
a global hard pool over continuous Detection/Caption scores.

Regular samples come from the remaining legal training population after E3, core, and
medium exclusions. Unselected easy mining candidates may be regular samples.

Outputs:

```text
data/processed/h2/core_hard_train.jsonl
data/processed/h2/medium_hard_train.jsonl
data/processed/h2/regular_representative_train.jsonl
data/processed/h2/h2_train.jsonl
data/processed/h2/h2_manifest.json
```

Medium/core metadata contains `h2_data_role`, hard score/reasons/diagnostics, cell rank,
cell identifier, and percentile. The manifest records all requested provenance,
allocation, distribution, score summaries, IDs, checks, and output hashes.

## H2-A Training Configuration

Both H2 configs strict-parse and enforce:

```text
start adapter            = Replay generalist
method                   = LoRA
LoRA r/alpha/dropout     = 16/32/0.05
vision_tuning.enabled    = false
freeze_vision_encoder    = true
loss                     = task_weighted, all weights 1.0
sampling_mode            = uniform
num_train_epochs         = null
max_steps                = null
target_effective_epochs  = 1.5
max_effective_epochs     = 2.0
allow_overtrain          = false
```

The 4090 profile uses BF16, batch size 16, accumulation 1, 12 workers, pinned memory,
and persistent workers. This follows the existing 4090 LoRA profile and was not newly
benchmarked on the current device. Vision stays frozen so H2-A tests the H2 data protocol
without mixing in the H1 visual adaptation variable.

## Modified and Added Files

Main additions:

- `src/sat_rs_vlm/evaluation/tier_builder.py`
- `src/sat_rs_vlm/training/refinement_dataset.py`
- `scripts/training/build_h2_mining_candidates.py`
- `scripts/training/build_h2_refinement_dataset.py`
- `configs/eval/evaluation_tiers_v2.yaml`
- `configs/eval/qwen3vl_eval_e1_v2.yaml`
- `configs/eval/qwen3vl_eval_e2_v2.yaml`
- `configs/eval/qwen3vl_eval_e3_v2.yaml`
- `configs/eval/qwen3vl_h2_mining.yaml`
- `configs/data/vrsbench_levircc_portable.yaml`
- `configs/train/qwen3vl_h2_global_refinement.yaml`
- `configs/train/qwen3vl_h2_global_refinement_4090.yaml`
- `docs/training/h2_global_multitask_refinement.md`
- v2 and H2 synthetic unit tests

Main integrations:

- additive full-population mode in the existing multisource preparer
- schema 2.0 branch in the canonical tier entrypoint
- tier version propagation through Evaluation v1.5 manifests
- version+SHA enforcement in paired comparison
- strict H2 config section in the existing training config
- default model-submission and AutoDL evaluation configs migrated to Unified E2 v2
- environment examples extended with explicit `REPLAY_ADAPTER_DIR`

No metric definition, bbox protocol, assistant-only masking, LoRA loading, visual
sidecar, Reliability fault behavior, checkpoint, prediction, or historical report was
modified.

## Tests

Executed in the requested default environment (`D:\APPS\ANACONDA\python.exe`):

```text
python -m compileall -q src scripts tests
PASS

new tier/H2/config/comparison/multisource targeted tests
30 passed

python -m pytest tests/unit -q
271 passed, 26 skipped, 1 warning

git diff --check
PASS
```

Skipped tests require optional PyTorch/model dependencies absent from the default local
environment. The warning is the pre-existing Pydantic `model_id` namespace warning.
Ruff is not installed in this environment. No GPU/model smoke, RTX 4090 benchmark,
6000-sample inference, or training was run.

## Next Commands

H2 mining and evaluation data remain on AutoDL. After the model, Replay adapter,
both datasets, and repository are available in the cloud workspace, run:

```bash
cd <AUTODL_PROJECT_ROOT>
export LOCAL_MODEL_DIR=<AUTODL_MODEL_DIR>
export REPLAY_ADAPTER_DIR=<AUTODL_REPLAY_ADAPTER>
export DATA_ROOT=<AUTODL_DATA_ROOT>
export OUTPUT_ROOT=<AUTODL_H2_OUTPUT>

python scripts/data/prepare_multisource_training_data.py \
  --config configs/data/vrsbench_levircc_portable.yaml \
  --full-evaluation-only

python scripts/evaluation/build_evaluation_tiers.py \
  --config configs/eval/evaluation_tiers_v2.yaml

python scripts/training/build_h2_mining_candidates.py \
  --config configs/train/qwen3vl_h2_global_refinement.yaml

python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_h2_mining.yaml \
  --checkpoint "$REPLAY_ADAPTER_DIR" \
  --output-dir reports/evaluation/h2_mining

python scripts/training/build_h2_refinement_dataset.py \
  --config configs/train/qwen3vl_h2_global_refinement.yaml \
  --evaluated-predictions reports/evaluation/h2_mining/evaluation_v1_5/evaluated_predictions.jsonl \
  --source-checkpoint "$REPLAY_ADAPTER_DIR"

python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_h2_global_refinement_4090.yaml \
  --dry-run

python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_h2_global_refinement_4090.yaml

python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml \
  --checkpoint "$OUTPUT_ROOT/checkpoints/lora/h2_global_refinement_4090"
```

本地只执行 `compileall`、unit tests 和 synthetic H2 builder tests；不下载或
运行 mining candidate 评测数据。
