# Training Data Statistics

- Samples: 125
- Supervision definition: `labels != -100`

## Task Distribution

| Task | Samples | Proportion |
|---|---:|---:|
| captioning | 25 | 0.2000 |
| counting | 25 | 0.2000 |
| detection | 25 | 0.2000 |
| scene_classification | 25 | 0.2000 |
| vqa | 25 | 0.2000 |

## Supervision Weighting Interpretation

- Task-level training weight is controlled by sampler draw frequency, not by supervised-token totals.
- `task_sampling_weights` are sampling weights, not loss weights.
- Model loss reduction: `mean over labels != -100 within each model call`.
- Supervised-token exposure below is a token-budget/truncation diagnostic, not an effective task-loss weighting table.

| Task | Population share | Estimated sampler draw share | Token exposure share |
|---|---:|---:|---:|
| captioning | 0.1423 | 0.1423 | 0.5137 |
| counting | 0.1084 | 0.1084 | 0.0168 |
| detection | 0.2550 | 0.2550 | 0.3775 |
| scene_classification | 0.0466 | 0.0466 | 0.0089 |
| vqa | 0.4477 | 0.4477 | 0.0831 |

## Truncation

- Max sequence length: 1024
- Truncated samples: 0
- Assistant-truncated samples: 0

## Detection Area Buckets

Thresholds: `{"small_max": 0.01, "medium_max": 0.1}`

## Notes

Visual token counts are processor-grid approximations. Unavailable fields are kept explicit.
