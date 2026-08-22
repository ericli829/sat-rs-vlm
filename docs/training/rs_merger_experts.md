# Task-Specialized RS Merger Experts

## Motivation

Object Adapter v0 asks whether frozen ViT taps can support a separate class-conditioned
counting/detection head. The merger-expert experiments ask a different question: can the normal
image + question -> textual answer VLM path improve when counting owns its visual organization and
compression parameters? There is no class ID input, detector/counting head, Hungarian matching, or
auxiliary numeric loss in C1/C2/C3. Supervision remains the repository's existing assistant-only
causal LM cross entropy.

The working hypothesis is that Qwen3-VL's shared main and DeepStack mergers are a multitask
information bottleneck. A specialist is allowed to optimize aggressively for one task because the
composite model keeps the original R1 route for every other task. Canonical `task_type`, rather than
prompt text, selects the route.

## Experiments

- C0 is the frozen formal R1 model on the original visual and language path.
- C1 clones the R1 main merger and all three DeepStack mergers after the R1 visual sidecar is
  loaded. Only the four clones train. It isolates the value of independent parameter space.
- C2 keeps the original R1 outputs and adds four independent zero-initialized RS detail residuals.
  It tests local spatial organization before fixed 2x2 compression.
- C3 adds exact-path counting-only LoRA to language layers 0-3 q/k/v/o. It tests whether shallow
  attention must adapt to the new visual distribution.

No variant adds visual tokens. Placeholder count, visual masks, mRoPE, context length, and
DeepStack positions stay identical to Qwen.

## DeepStack injection and first consumption

The runtime audit proves the order from the installed Transformers class source and records exact
named paths. Qwen first scatters the final merger output into input embeddings, so decoder layer 0
consumes it first. Inside the language loop, a decoder layer runs before `_deepstack_process`:

- final ViT output -> main merger -> first consumed by language layer 0;
- first configured ViT tap -> DeepStack 0, injected after layer 0 -> first consumed by layer 1;
- second tap -> DeepStack 1, injected after layer 1 -> first consumed by layer 2;
- third tap -> DeepStack 2, injected after layer 2 -> first consumed by layer 3.

The formal 4B contract expected by this experiment is 24 vision blocks, vision hidden size 1024,
LLM hidden size 2560, merge size 2, and taps `[5, 11, 17]`. Any runtime difference blocks training
and is written to `reports/rs_merger_expert/source_architecture_audit.*`. The audit also executes a
raw `[4,1024]` input probe through every base merger and requires main-merger norm shape `[1024]`,
DeepStack norm shape `[4096]`, packed hidden/FC input width 4096, and output width 2560. A merger
that only accepts an already packed input, or exposes inconsistent norm/linear shapes, fails closed.

## C2 architecture and Qwen ordering

Each tap owns its complete branch; no weights are shared.

The residual inputs are captured directly from ViT block outputs, before calling any merger:
F5/F11/F17 feed residual branches 0/1/2 and F23 feeds the final residual branch. The merger wrapper
input is used only by the frozen base merger. This separation is intentional: even if a runtime
passes a 2x2-packed 4096-d tensor to a DeepStack merger wrapper, the detail branch still receives
the corresponding raw 1024-d block output.

```text
raw ViT [K,1024]
  -> LayerNorm(1024)
  -> Linear(1024,512)
  -> unpack Qwen block-major tokens to [T,H,W,512]
  -> X + PWConv1x1(GELU(DWConv3x3(X)))
  -> repack raster grid to Qwen block-major order
  -> contiguous 2x2 space-to-depth [K/4,2048]
  -> ZERO initialized Linear(2048,2560)
  -> DeltaZ

Z_count = Z_R1 + DeltaZ
```

The Qwen processor orders patches as
`T,H_block,W_block,H_inner,W_inner`; a plain H-major reshape is wrong. The implementation exposes
`unpack_to_spatial_grid()` and `repack_qwen_merge_order()` and tests position-coded round trips and
token counts across dynamic/multi-image grids.

## Frozen foundation mode and accumulation

Training keeps the frozen ViT, frozen language model, and frozen base mergers in evaluation mode.
Only cloned/detail expert modules and Count Interface LoRA stochastic layers enter training mode,
so foundation dropout stays disabled while Count LoRA dropout remains active. Evaluation mode does
not disable PyTorch autograd; no `no_grad` boundary is placed around the frozen language model, and
the assistant-only LM loss therefore still backpropagates through it into the merger expert.

Gradient accumulation is bounded by each data-loader epoch. A full window divides each microbatch
loss by the configured accumulation count; a short epoch-tail window divides by its actual size and
always performs an optimizer step. Consequently a one-effective-epoch plan covers every sample and
uses `ceil(number_of_microbatches / accumulation_steps)` optimizer steps, including non-divisible
tails.

## C3 interface LoRA

Only language attention layers 0-3 q/k/v/o are wrapped. Those are the first layers that consume the
main and three DeepStack visual streams. The rank is 16, alpha 32, and dropout 0.05; `lora_B` is zero
initialized. RMSNorm and MLP projections remain frozen. Exact resolved paths are used, so vision
attention and language layers 4+ cannot match accidentally. The base route bypasses Count LoRA.

R1 is preferably merged with `safe_merge=True` only after fixed-probe generation, logits, and all
four merger outputs prove parity. If merge is unavailable or fails parity, the loader discards the
mutated object, reloads one clean frozen R1 PEFT model, reapplies the sidecar, and uses an additive
exact-path Count LoRA. It never duplicates a 4B LLM in the saved checkpoint.

## Composite checkpoints

`checkpoint/expert_model.safetensors` contains only cloned-merger or detail-branch tensors plus
Count LoRA tensors. `expert_manifest.json` records R1, sidecar, data, architecture audit hashes,
runtime versions, dimensions, parameter counts, initialization, and routing. The loader requires an
exact state-key set and exact provenance/architecture values; partial loading is an error.

`--resume-from-checkpoint` currently restores composite expert weights but intentionally resets the
optimizer and scheduler, and records that fact in preflight. Do not use it as an exact interrupted-run
resume until optimizer/scheduler state checkpointing is added.

To add a future Detection Expert, register another canonical task route and another independent
four-tap expert state under the same controller/checkpoint contract. No learned router or prompt
regex is needed.

## Environment

```bash
export PROJECT_ROOT=/root/autodl-tmp/sat-rs-vlm
export MODEL_ROOT=/root/autodl-tmp/models
export DATA_ROOT=/root/autodl-tmp/datasets/VRSBench
export OUTPUT_ROOT=/root/autodl-tmp/outputs
export R1_CHECKPOINT=/root/autodl-tmp/outputs/<formal-r1>/r1/adapter
export R1_VISUAL_SIDECAR=$R1_CHECKPOINT/visual_trainable_weights.safetensors
cd "$PROJECT_ROOT"
```

## Data and architecture audit

```bash
python scripts/training/build_rs_count_merger_data.py \
  --source-train data/processed/qwen3vl_train.jsonl \
  --source-manifest data/evaluation/tiers/evaluation_tiers_manifest.json \
  --output-dir data/processed/rs_count_merger_v1

python scripts/training/audit_rs_merger_architecture.py \
  --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" \
  --r1-checkpoint "$R1_CHECKPOINT" \
  --visual-sidecar "$R1_VISUAL_SIDECAR"
```

## C1 commands

```bash
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c1_clone_4090.yaml --dry-run
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c1_clone_4090.yaml --max-train-samples 64 --max-steps 2 --skip-eval
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c1_clone_4090.yaml --skip-eval

python scripts/evaluation/evaluate_rs_merger_expert.py --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" --r1-checkpoint "$R1_CHECKPOINT" --visual-sidecar "$R1_VISUAL_SIDECAR" --architecture-audit reports/rs_merger_expert/source_architecture_audit.json --expert-checkpoint "$C1_CHECKPOINT" --tier-file data/evaluation/tiers/e1_quick.jsonl --image-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/rs_merger_expert/c1_e1" --baseline-metrics "$C0_E1_METRICS"
python scripts/evaluation/evaluate_rs_merger_expert.py --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" --r1-checkpoint "$R1_CHECKPOINT" --visual-sidecar "$R1_VISUAL_SIDECAR" --architecture-audit reports/rs_merger_expert/source_architecture_audit.json --expert-checkpoint "$C1_CHECKPOINT" --tier-file data/evaluation/tiers/e2_standard.jsonl --image-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/rs_merger_expert/c1_e2" --baseline-metrics "$C0_E2_METRICS"
```

For the new evaluator's C0/base-route probe, add `--force-base` to either evaluation command. Its
prediction/metric output must match the original formal R1 evaluator on the same frozen tier.

## C2 commands

Replace the C1 config with `rs_count_merger_c2_detail_4090.yaml` for dry-run, 64-sample/two-step
smoke, and formal training. Evaluate with the same commands while replacing `$C1_CHECKPOINT` with
`$C2_CHECKPOINT` and the output roots with `c2_e1`/`c2_e2`.

```bash
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c2_detail_4090.yaml --dry-run
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c2_detail_4090.yaml --max-train-samples 64 --max-steps 2 --skip-eval
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c2_detail_4090.yaml --skip-eval

python scripts/evaluation/evaluate_rs_merger_expert.py --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" --r1-checkpoint "$R1_CHECKPOINT" --visual-sidecar "$R1_VISUAL_SIDECAR" --architecture-audit reports/rs_merger_expert/source_architecture_audit.json --expert-checkpoint "$C2_CHECKPOINT" --tier-file data/evaluation/tiers/e1_quick.jsonl --image-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/rs_merger_expert/c2_e1" --baseline-metrics "$C0_E1_METRICS"
python scripts/evaluation/evaluate_rs_merger_expert.py --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" --r1-checkpoint "$R1_CHECKPOINT" --visual-sidecar "$R1_VISUAL_SIDECAR" --architecture-audit reports/rs_merger_expert/source_architecture_audit.json --expert-checkpoint "$C2_CHECKPOINT" --tier-file data/evaluation/tiers/e2_standard.jsonl --image-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/rs_merger_expert/c2_e2" --baseline-metrics "$C0_E2_METRICS"
```

## C3 commands

Replace C2 with `rs_count_merger_c3_detail_lora_4090.yaml`. Evaluation uses `$C3_CHECKPOINT` and
`c3_e1`/`c3_e2` output roots.

```bash
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c3_detail_lora_4090.yaml --dry-run
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c3_detail_lora_4090.yaml --max-train-samples 64 --max-steps 2 --skip-eval
python scripts/training/train_rs_merger_expert.py --config configs/experiments/rs_count_merger_c3_detail_lora_4090.yaml --skip-eval

python scripts/evaluation/evaluate_rs_merger_expert.py --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" --r1-checkpoint "$R1_CHECKPOINT" --visual-sidecar "$R1_VISUAL_SIDECAR" --architecture-audit reports/rs_merger_expert/source_architecture_audit.json --expert-checkpoint "$C3_CHECKPOINT" --tier-file data/evaluation/tiers/e1_quick.jsonl --image-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/rs_merger_expert/c3_e1" --baseline-metrics "$C0_E1_METRICS"
python scripts/evaluation/evaluate_rs_merger_expert.py --base-model "$MODEL_ROOT/Qwen3-VL-4B-Instruct" --r1-checkpoint "$R1_CHECKPOINT" --visual-sidecar "$R1_VISUAL_SIDECAR" --architecture-audit reports/rs_merger_expert/source_architecture_audit.json --expert-checkpoint "$C3_CHECKPOINT" --tier-file data/evaluation/tiers/e2_standard.jsonl --image-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/rs_merger_expert/c3_e2" --baseline-metrics "$C0_E2_METRICS"
```

Do not begin formal training unless the runtime audit, R1 merge/additive parity, step-0 parity,
ordering tests, trainable audit, and real two-step CUDA smoke all pass.
