# Unified-v2 evaluation tiers and post-training recovery

The project keeps three identities separate:

1. **Formal R1 reference tier**: the frozen R1 E1/E2 bytes recorded by
   `formal_r1_sha256`. They are provenance references, not generated Unified-v2
   outputs.
2. **Generated Unified-v2 tiers**: E1/E2/E3 built deterministically from the
   configured VRSBench + LEVIR-CC population. They should live under an external
   runtime `EVAL_TIER_ROOT`.
3. **E_COUNT_V2**: a derived tier containing every Unified-v2 E2 counting row
   plus every non-counting Unified-v2 E1 guard row. It retains raw counting rows;
   the exact-cardinality protocol determines the 327 eligible rows.

Raw SHA-256 records byte identity. `canonical_jsonl_sha256` records semantic
JSONL identity while preserving row order. A canonical mismatch is a hard fail.
Raw drift with a matching canonical hash is reported as provenance drift and is
allowed for the new manifest schema. Legacy manifests remain raw-SHA hard gates.

## Local build

```powershell
$env:EVAL_DATA_ROOT = 'F:\VIT-data'
$env:EVAL_TIER_ROOT = 'D:\Desktop\tzb-2026\outputs\evaluation_tiers\unified_v2'
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' `
  scripts/evaluation/prepare_unified_v2_bundle.py `
  --config configs/eval/evaluation_tiers_v2.yaml `
  --output-root $env:EVAL_TIER_ROOT
```

The command builds in a sibling temporary directory and swaps the complete
bundle only after all invariants pass. If the newly generated E_COUNT_V2 sample
IDs/order or row semantics differ from the tracked benchmark, it stops and
writes a migration diff for manual approval. For the approved multisource
population migration (same IDs/order and 327 exact-cardinality rows, with the
population's updated prompt wording), rerun with the explicit flag:

```bash
python scripts/evaluation/prepare_unified_v2_bundle.py \
  --config configs/eval/evaluation_tiers_v2.yaml \
  --output-root "$EVAL_TIER_ROOT" \
  --allow-benchmark-migration
```

## AutoDL build and recovery

```bash
cd /root/autodl-tmp/sat-rs-vlm
export EVAL_TIER_ROOT=/root/autodl-tmp/outputs/evaluation_tiers/unified_v2
export E_COUNT_V2_FILE="$EVAL_TIER_ROOT/e_count_v2.jsonl"
export E_COUNT_V2_MANIFEST="$EVAL_TIER_ROOT/e_count_v2_manifest.json"
export EVAL_DATA_ROOT=/root/autodl-tmp/datasets
python scripts/evaluation/prepare_unified_v2_bundle.py \
  --config configs/eval/evaluation_tiers_v2.yaml \
  --output-root "$EVAL_TIER_ROOT"
```

If training finished but fixed evaluation failed, finalize the existing run;
this executes no optimizer step and does not retrain:

```bash
python scripts/training/train_rs_merger_expert.py \
  --config configs/experiments/rs_count_merger_c2_lm_4e.yaml \
  --finalize-existing-run \
  /root/autodl-tmp/outputs/counting_expert/four_epoch_matrix/C2_LM_4E_20260823_035157
```

For this run, set `DATA_ROOT=/root/autodl-tmp/datasets/VRSBench` for the
training JSONL's `Images/...` paths; keep `EVAL_DATA_ROOT` at the common
`/root/autodl-tmp/datasets` parent so both VRSBench and LEVIR-CC fixed-eval
paths resolve.

The recovery path validates `training_state.pt`, all completed epoch sidecars,
and the target epoch, evaluates each epoch with the configured E_COUNT_V2
manifest, writes the learning curve, and produces the final composite
checkpoint. The final expert tensors are checked against the last epoch sidecar.

Use `--skip-completed` with the serial launcher to skip only runs that contain
both final checkpoint artifacts and validated R1/visual provenance. A run with
epoch sidecars but no final checkpoint is not complete; finalize it first.
