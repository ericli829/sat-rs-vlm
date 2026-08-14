# Evaluation Tiers

Formal model submission evaluation uses the frozen **E2 standard tier** by
default. E1 and E3 are explicit alternatives:

| Tier | Meaning | Typical use |
|---|---|---|
| E1 | Quick diagnostic set, about 593 samples | Debugging and checkpoint screening |
| E2 | Standard stratified set, about 3000 samples | Every completed training run and model submission |
| E3 | Full legal evaluation population | Final candidate, paper, quantization, or reliability conclusions |

The fixed assets are generated with:

```powershell
python scripts/evaluation/build_evaluation_tiers.py `
  --config configs/eval/evaluation_tiers.yaml
```

The generator writes `data/evaluation/tiers/e1_quick.jsonl`,
`e2_standard.jsonl`, `e3_full.jsonl`, and
`evaluation_tiers_manifest.json`. Evaluation refuses to run when the selected
tier file is missing, absent from the manifest, or has a different SHA256.

## Submission Commands

Default E2 evaluation:

```powershell
python scripts/evaluate_rs_vlm.py `
  --config configs/eval/qwen3vl_eval.yaml `
  --checkpoint checkpoints/<experiment>
```

The equivalent explicit commands are `make eval-e1`, `make eval-e2`, and
`make eval-e3`. Paired comparison records the tier and tier SHA256; baseline
and candidate results from different tiers cannot be compared.

The AutoDL full pipeline uses `configs/cloud/evaluate_lora_autodl.yaml`, which
also points to E2. The `run_v15_sensitivity.py` Reliability sweep intentionally
uses E1 for broad fault-condition screening; it is a specialized diagnostic,
not a model submission result. Promote a high-risk condition to E2/E3 only by
changing its Reliability experiment configuration explicitly.

Legacy dataset-specific configs and smoke configs remain available for
diagnostics. They are not formal submission evaluation inputs and must not be
used for headline model comparisons.
