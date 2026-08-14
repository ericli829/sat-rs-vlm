# Historical ViT Training Recovery Precheck

## Git history

- Current branch: `master`
- Current HEAD: `8b23eb6` (`origin/master`)
- Historical common point: `2d8504e`
- Historical ViT tip: `fe7fd466`
- Manual integration commit: `5ca9b2f`
- `5ca9b2f` has one parent (`2d8504e`), so it is not a two-parent Git merge.

## Audit table

| Historical path | Current status | Action | Reason |
|---|---|---|---|
| `src/sat_rs_vlm/training/losses.py` | Missing | Restore and test | Canonical token-mean/task-weighted loss strategy is absent. |
| `src/sat_rs_vlm/training/trainer.py` | Missing | Restore and compose sampler/sidecar support | The current sampler-local Trainer cannot execute configured multitask loss. |
| `src/sat_rs_vlm/training/data_statistics.py` | Missing | Restore | Exact assistant-mask and truncation statistics have no current source implementation. |
| `src/sat_rs_vlm/training/hard_example_mining.py` | Missing | Restore, then expose reusable scoring/categorization helpers | H1 reproducibility and future H2 data construction depend on it. |
| `src/sat_rs_vlm/training/loss_diagnostics.py` | Missing | Restore | Read-only loss-length diagnostics still have historical value. |
| `scripts/training/analyze_training_data.py` | Missing | Restore | Formal statistics CLI. |
| `scripts/training/build_hard_example_dataset.py` | Missing | Restore | Formal H1 experiment entrypoint; a thin CLI is intentional. |
| `scripts/training/estimate_h1_steps.py` | Missing | Restore | Keeps H1 budget selection explicit. |
| `scripts/debug/diagnose_multitask_loss_bias.py` | Missing | Restore | Read-only real-model diagnostic; never changes Trainer behavior. |
| `src/sat_rs_vlm/training/config.py` | Partial | Merge historical loss/statistics/hard models with current dynamic-plan/vision fields; forbid unknown fields | Current YAML keys are silently ignored. |
| `src/sat_rs_vlm/data/qwen3vl_collator.py` | Partial | Restore task metadata and exact tokenization diagnostics without changing assistant-only mask | Multitask Trainer and statistics need metadata/diagnostics. |
| `scripts/train_qwen3vl_lora.py` | Partial | Integrate MultitaskTrainer, H1 visual audit/groups/sidecars, and retain dynamic plan | This is the execution path that must prove the configured loss is active. |
| `src/sat_rs_vlm/data/task_sampler.py` | Duplicate Trainer construction | Keep sampler builders; remove the embedded Trainer implementation | Trainer composition belongs in one canonical implementation. |
| `src/sat_rs_vlm/training/training_plan.py` | Current-only canonical implementation | Preserve unchanged | Protects effective epochs and replaces fixed default step budgets. |
| `src/sat_rs_vlm/training/optimizer.py` | Newer current implementation | Preserve; call generic grouped optimizer APIs | It validates non-overlapping LoRA/merger/ViT groups. |
| `src/sat_rs_vlm/training/vision_tuning.py` | Newer current implementation | Preserve; reconnect from training entry | Uses generic sidecar name and manifest/checksum validation. |
| Historical training tests | Missing or partial | Restore only tests covering recovered behavior; adapt to current APIs | Prevents another silent integration loss. |
| `evaluation/visual_analysis.py`, CLI and test | Missing while H1 documentation still references them | Restore as an additive diagnostic | It consumes canonical evaluated predictions and does not create a second Evaluation metric pipeline. |
| E1/E2/E3 JSONL and manifest | Current canonical frozen assets | Do not modify or regenerate | Sample IDs and historical comparisons must remain stable. |
| Reliability modules/tests | Current canonical implementation | Do not modify | Training recovery must remain independent of Reliability APIs. |

## Compatibility risks

1. Pydantic `extra=forbid` may expose stale YAML keys; every failure must be resolved explicitly.
2. Transformers Trainer signatures vary by version; the historical compatibility wrapper must remain version tolerant.
3. PEFT can freeze base parameters after adapter load; visual unfreeze must happen after `is_trainable=True` adapter loading.
4. A grouped optimizer must include every trainable tensor exactly once and must coexist with Trainer scheduler construction.
5. H1 visual sidecars must use the current generic filename/manifest contract, not the historical filename.
6. Evaluation tiers and Reliability are outside the edit surface and will be regression-tested.
