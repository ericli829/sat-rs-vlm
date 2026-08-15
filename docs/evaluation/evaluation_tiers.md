# Evaluation Tiers

The repository has two immutable tier generations. A tier name alone is not a complete
evaluation identity; every result records `tier_version` and tier SHA256.

| Generation | Scope | Directory | Use |
|---|---|---|---|
| `legacy-vrs-v1` | VRSBench only | `data/evaluation/tiers/` | Historical Stage A/B, Replay, H1, quantization and Reliability comparisons |
| `unified-v2` | VRSBench + LEVIR-CC | `data/evaluation/tiers_v2/` | Future formal training and model submissions |

Legacy JSONL files are frozen and must never be regenerated. The explicit configs
`qwen3vl_eval_e1.yaml`, `qwen3vl_eval_e2.yaml`, and `qwen3vl_eval_e3.yaml` continue to
address those historical assets. `qwen3vl_eval.yaml` and the three `_v2` configs address
Unified v2; E2 v2 is the default.

## Unified Population

Unified E3 is the complete legal evaluation population:

- VRSBench: every row in the complete validation JSONL.
- LEVIR-CC: one deterministic reference case per validation image pair. The source has
  five captions per pair; repeated captions for the same pair are not five independent
  evaluation cases under the current canonical `validation_group_by_images=true` policy.

Every image path is relative to common `${DATA_ROOT}`, for example
`VRSBench/Images/...` or `LEVIR-CC/images/...`. Population preparation and tier building
fail with sample ID, dataset, and path when an image cannot be resolved.

Prepare the complete portable population on AutoDL:

```bash
python scripts/data/prepare_multisource_training_data.py \
  --config configs/data/vrsbench_levircc_portable.yaml \
  --full-evaluation-only
```

Build frozen Unified tiers:

```bash
python scripts/evaluation/build_evaluation_tiers.py \
  --config configs/eval/evaluation_tiers_v2.yaml
```

The selection is deterministic with seed 42 and guarantees `E1 < E2 < E3`. E1/E2 use
80% VRSBench and 20% LEVIR-CC when capacity permits. VRSBench task allocation follows
`sqrt(N_task)` softened balancing; task-specific subtype allocation favors diagnostic
coverage. E3 retains the natural full-population distribution.

## Commands

Default Unified E2 v2:

```bash
python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml \
  --checkpoint checkpoints/<experiment>
```

Explicit Unified tiers use `qwen3vl_eval_e1_v2.yaml`, `qwen3vl_eval_e2_v2.yaml`, and
`qwen3vl_eval_e3_v2.yaml`. Use the legacy configs only when comparing to an existing
legacy result. Paired comparison rejects different versions or SHA256 values, including
`legacy-vrs-v1/E2` versus `unified-v2/E2`.

E1 is for quick diagnostics, E2 for every completed formal run, and E3 for final
candidate, paper, deployment quantization, or Reliability conclusions. Dataset-specific
VRSBench and LEVIR-CC configs remain useful diagnostics but are not a unified headline
result.
