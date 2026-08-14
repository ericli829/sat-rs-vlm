# Multitask Loss Length-Bias Diagnostic

- Mixed batches: 20
- Executed samples: 40
- Supervised assistant tokens: 921
- Current batch loss mean: 0.760039
- Per-sample normalized loss mean: 0.678983
- Peak CUDA allocated memory (MB): 5311.02685546875
- Peak CUDA reserved memory (MB): 7144.0

## Task Shares

| Task | Sample share | Supervised-token share | Loss-numerator share | Mean token CE | Mean sample loss |
|---|---:|---:|---:|---:|---:|
| captioning | 0.2000 | 0.6515 | 0.7082 | 1.087183 | 1.074846 |
| counting | 0.2000 | 0.0261 | 0.0277 | 1.063881 | 1.063881 |
| detection | 0.2000 | 0.2595 | 0.2546 | 0.981181 | 0.984292 |
| scene_classification | 0.2000 | 0.0337 | 0.0060 | 0.178031 | 0.165828 |
| vqa | 0.2000 | 0.0293 | 0.0035 | 0.120350 | 0.106066 |

## Judgement

- Material share-gap threshold: 0.1000
- Length-associated overrepresented tasks: captioning
- Captioning/detection overrepresented: captioning
- VQA/counting underrepresented: vqa, counting
- Per-sample normalized-loss experiment supported: True
- Conclusion: Evidence supports a per-sample normalized-loss experiment: long-answer tasks are materially overrepresented in the batch loss numerator while VQA/counting are underrepresented.

## Interpretation

The normal Causal LM batch loss is the mean CE across all assistant tokens in the mixed batch. `loss_numerator_share` therefore tests token-level influence before the common batch denominator is applied. This is a read-only diagnostic; it does not change the formal H1 loss or Trainer.
