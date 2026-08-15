# H2 Global Multitask Refinement

## Goal

H2 improves the complete remote-sensing task portfolio while limiting regression on any
core task. It starts from the **Replay generalist adapter**, not H1. H1 remains a visual
specialist candidate.

```text
Replay Generalist
  -> build H2 mining candidates
  -> AutoDL Replay inference
  -> Evaluation v1.5 evaluated_predictions
  -> build H2 final dataset
  -> AutoDL H2-A LoRA
  -> Unified E2 v2 evaluation
```

H2 mining is a cloud-side workflow. The candidate images, Replay adapter,
Evaluation v1.5 outputs, and final H2 JSONL remain on AutoDL; they are not
downloaded to the local workstation. Local execution is limited to code checks
and synthetic/unit tests.

H2-A changes only data composition and continued LoRA optimization. Vision remains
frozen, LoRA rank and bbox protocol remain unchanged, sampling is uniform, and the loss
is per-sample `task_weighted` with all task weights equal to 1.0.

## Data Protocol

Mining candidates default to 6000 samples with VRSBench/LEVIR-CC source weights 75/25.
VRSBench task allocation uses `sqrt(N_task)` softened balancing. Unified E3 v2 IDs are
always excluded.

The Replay checkpoint predicts all candidates, and Evaluation v1.5 produces the only
difficulty input. The builder does not load a model and does not implement duplicate
metrics.

Final H2 defaults to 8000 samples:

| Role | Share | Meaning |
|---|---:|---|
| `regular_representative` | 60% | Representative replay from the remaining legal population |
| `medium_hard` | 25% | The next difficult candidates within each source/task cell |
| `core_hard` | 15% | The top difficult candidates within each source/task cell |

Source balance and difficulty balance are independent dimensions. Core and medium are
ranked separately inside each `dataset × task_type` cell using `hard_score DESC, id ASC`.
This is essential because VQA/Scene scores are often discrete while Detection/Caption
scores are continuous. A global score threshold would allow one task's numeric scale to
dominate the hard pool. Optional thresholds remain experimental and are not the H2
default protocol.

Each output row has `metadata.h2_data_role`. Medium/core rows also retain hard score,
reasons, diagnostics, cell rank, cell name, and percentile. `h2_manifest.json` records
input/output hashes, protected E3 identity, allocation targets and actuals, all selected
IDs, and duplicate/leakage checks.

## AutoDL Linux: mining and training

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
  --checkpoint <AUTODL_H2_OUTPUT>/checkpoints/lora/h2_global_refinement_4090
```

The 4090 profile uses BF16, micro-batch size 4, accumulation 4, 12 workers, pinned
memory, and persistent workers. The effective batch size remains 16 while the smaller
micro-batch avoids the observed 24 GB VRAM overflow.

## Training and Guardrails

H2-A uses dynamic training length:

```text
target_effective_epochs = 1.5
max_effective_epochs    = 2.0
max_steps               = null
allow_overtrain         = false
```

The preferred outcome relative to Replay is Scene, VQA, Counting, and Detection flat or
better, with no meaningful Caption or LEVIR regression. A provisional diagnostic
guardrail is no core task declining by more than two percentage points, but this must be
calibrated from repeated E1/E2 variance and is not hard-coded into training.

H2-B may later copy this config and separately enable merger-only or merger plus last-1
ViT. Existing vision tuning, grouped optimizer, and visual sidecar code already supports
that experiment; H2-B is not part of H2-A.
