# H1 Visual Adaptation Precheck

Generated: 2026-08-11

## Current training entrypoint

- The supported entrypoint is `scripts/train_qwen3vl_lora.py`.
- It loads one `Qwen3VLDataset`, batches with `Qwen3VLDataCollator`, and creates a
  Transformers `Trainer` through `sat_rs_vlm.data.task_sampler.create_trainer`.
- Existing LoRA continuation is already supported through
  `lora.initial_adapter_dir` or `--initial-adapter`. The adapter is loaded with
  `PeftModel.from_pretrained(..., is_trainable=True)`.
- The existing LoRA and QLoRA modes share this entrypoint. H1 will remain a LoRA
  continuation mode and will not alter QLoRA behavior.

## Assistant-only loss status

- `Qwen3VLDataCollator` builds the full conversation and generation-prompt encodings
  with the same processor and chat template.
- `_build_assistant_labels` masks prompt/image-placeholder tokens and padding with
  `-100`; only the remaining assistant answer tokens are supervised.
- It raises an error when truncation leaves no assistant tokens.
- H1 must reuse this Collator unchanged. The loss mask is already correct and is not
  part of the planned modification.

## Current trainable-parameter behavior

- The current training path optionally freezes vision/projector parameters before
  PEFT wrapping, then injects or loads LoRA.
- Vision freezing is keyword based and cannot express "last N visual blocks".
- There is no complete trainable-parameter audit. Only aggregate trainable/total
  counts are printed.
- The current Trainer uses one global learning rate. Task sampling weights affect
  sampling probability only; they are not loss weights.

## Current PEFT adapter load behavior

- Existing checkpoints remain compatible because adapter continuation uses the
  standard PEFT `adapter_config.json` and `is_trainable=True`.
- The initial adapter and output directory are required to differ, preventing an H1
  run from overwriting its starting checkpoint.
- H1 will load PEFT first, freeze all non-LoRA base parameters, re-enable adapter
  parameters, then explicitly unfreeze selected visual modules before optimizer
  construction.

## Actual Qwen3-VL visual structure

The local model at `D:/Desktop/tzb-2026/Qwen3-VL-2B-Instruct` was inspected without
loading tensors into GPU memory. `config.json`, installed Transformers source, and
the 625 safetensors parameter keys agree on the following structure:

- Vision depth: 24 blocks.
- Checkpoint block keys: `model.visual.blocks.0` through
  `model.visual.blocks.23`.
- Main merger: `model.visual.merger`.
- Deep-stack mergers: `model.visual.deepstack_merger_list.0` through `.2`.
- Patch embedding: `model.visual.patch_embed`.
- Runtime model structure: `Qwen3VLForConditionalGeneration.model.visual`, whose
  visual module owns `blocks`, `merger`, `deepstack_merger_list`, and `patch_embed`.
- PEFT may prepend wrapper names such as `base_model.model`; implementation must
  resolve modules by object structure and parameter identity, not a single fixed
  full parameter-name prefix.

H1 default selection is therefore runtime blocks 22 and 23 plus the main merger.
The implementation will still derive this from `len(visual.blocks)` and will not
hard-code 24, 22, or 23.

## Reusable configuration and evaluation components

- `Qwen3VLTrainingConfig` already centralizes model, data, LoRA, training,
  evaluation, and logging configuration and expands `${ENV}` paths.
- `initial_adapter_dir`, `max_steps`, seed, scheduler, checkpoint cadence, and
  gradient settings can be reused directly.
- Evaluation v1.5 is the sole formal evaluation implementation under
  `sat_rs_vlm.evaluation`. Its `evaluated_predictions.jsonl` already includes
  per-sample parse status, IoU, generalized IoU, center distance, counting error,
  normalized text scores, ROUGE-L/chrF, and CIDEr-D approximation.
- Hard mining will consume those evaluated rows instead of defining a parallel
  evaluator.

## Planned changed files

- Extend `src/sat_rs_vlm/training/config.py` with optional H1 statistics, mining,
  vision-tuning, grouped-optimization, and audit models while preserving old YAML.
- Add `training/data_statistics.py`, `training/hard_example_mining.py`,
  `training/vision_tuning.py`, and `training/optimizer.py`.
- Extend `data/task_sampler.py` with a formal Trainer subclass/factory path for an
  explicitly supplied optimizer, without runtime monkey patching.
- Extend `scripts/train_qwen3vl_lora.py` only at stable extension points for H1
  preparation, auditing, grouped optimizer construction, and reporting.
- Add training CLIs for statistics, hard dataset construction, and H1 step
  estimation.
- Add an Evaluation v1.5 visual-analysis module/CLI for bbox-size and visual-token
  correlation reports.
- Add `configs/train/qwen3vl_hard_visual_adaptation.yaml`, unit/integration tests,
  H1 documentation, and the final integration report.

## Compatibility risks

1. PEFT wrappers add name prefixes. Classification must use module/parameter
   identity and suffix diagnostics, not assume raw checkpoint names.
2. A model loaded with `device_map=auto` may span devices. H1 must retain the
   existing device-safe batch movement and avoid moving the whole model after load.
3. Visual blocks and merger are base parameters, so adapter-only checkpoint saving
   is insufficient by itself. H1 must save those trained base-module weights in a
   sidecar artifact and record them in the strategy manifest while preserving the
   PEFT adapter format.
4. Trainer normally creates one optimizer from all trainable parameters. H1 needs
   an explicit optimizer with non-overlapping LoRA/merger/ViT groups supplied before
   training.
5. Training-source metadata is not uniform across all historical JSONL files.
   Statistics/mining must use documented fallbacks and report unknown values rather
   than inventing provenance.
6. Truncation cannot be inferred from a truncated tensor alone. Statistics must
   compare the normal capped encoding with an uncapped/no-truncation encoding made
   through the same processor and chat template.
7. The fixed 593-sample evaluation set must be supplied as an ID file or Evaluation
   v1.5 output and checked before writing H1 data. Missing exclusion evidence must
   fail closed for production mining.

