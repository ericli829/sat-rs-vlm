# UHR Locator experiment plan

This file records planned ablations and the small visual diagnostic workflow.
The checked-in defaults are starting points; they are not benchmark-tuned
values. Development sweeps should use a development population. The current
five-image MME manifest uses answer-labelled benchmark samples and is therefore
explicitly `DIAGNOSTIC ONLY; NOT FOR FINAL HYPERPARAMETER TUNING`.

## Controlled variants

| ID | Locator evidence/search | Purpose |
|---|---|---|
| A | retrieval only | Establish question-to-region relevance baseline |
| B | detector only | Measure object-proposal coverage without semantic retrieval |
| C | detector + retrieval | Test complementary object and full-query evidence |
| D | C + deterministic spatial prior | Measure value of parsed image-part constraints |
| E | D + adaptive hierarchy | Compare multiscale search with a fixed/full grid budget |
| F | VisRAG-Ret vs lightweight retriever | Measure relevance quality, latency, and memory tradeoff |
| G | rules parser vs future small-LM fallback | Measure routing gains without changing providers |

Keep the image population, answer model, provider checkpoint, final ROI budget,
fusion settings, and evaluation protocol fixed unless the row explicitly changes
one of them. Always report provider/model provenance and the complete search
configuration.

## Metrics

- **Coverage@K:** fraction of target reference geometry covered by the top K
  final regions.
- **Target Recall:** fraction of annotated targets intersecting at least one
  selected region at the declared coverage threshold.
- **Processed Area Ratio:** cumulative scored-view area divided by original
  image area. It may exceed one for multilevel/halo inspection.
- **Selected Union Area Ratio:** geometric union of final evidence ROIs divided
  by image area; it is bounded by one.
- **Processed Union Area Ratio:** geometric union of every scored view divided
  by image area; it is bounded by one.
- **Coverage / Area:** target coverage normalized by search area.
- **Latency:** parser, detector, retrieval, search/fusion, and end-to-end time.
- **Crop Count:** scored views and final Multi-ROI count, reported separately.
- **Peak Memory:** provider and total CUDA peak when available.
- **Final VQA Accuracy:** answer metric using a fixed downstream VLM and prompt.

Also retain failure counts, parser warnings, stop reasons, selected depth,
provider cache hits, and per-component raw/normalized score traces. Counting
experiments should separately report global-coordinate object deduplication
quality; they must not sum per-crop counts.

## Suggested sequence

1. Validate A/B/C on a development population using the same maximum crop and
   cumulative-area budgets.
2. Add D and inspect whether spatial priors improve coverage without merely
   reducing explored area.
3. Compare E against a fixed-depth, fixed-beam control at similar computation.
4. Run F with identical candidate regions and cache policy before end-to-end
   comparison.
5. Attempt G only after a deterministic parser-error taxonomy is available; the
   learned fallback must expose confidence and preserve rules-first behavior.

## Staged visual diagnostics

`configs/experiments/uhr_locator/diagnostic_v1.yaml` defines 16 named presets,
not a Cartesian product. Each family inherits the same E0 mechanics and changes
only halo, tiled detector geometry, standardized-beam behavior, zoom target, or
final context margin. Scorer weights remain fixed in the first pass.

The five-sample manifest covers a prominent airport, dense airplane counting,
distributed ships, a pier/bank relation, and a fine harbor-building attribute.
Each image is 2K--4.4K. No GT boxes are invented; coverage fields remain absent
and the generated human-review fields stay blank.

Every completed sample writes `result.json`, `search_trace.json`, per-depth
search overlays, a final ROI overlay, detector proposals with tile provenance,
individual crops, and a contact sheet containing the question and reference
answer. The run root writes its config/sample hashes, `summary.csv`,
`summary.md`, and `human_review.csv`. A provider initialization or inference
failure writes `failure.json` and stops; the runner never silently substitutes a
mock provider.

Remote Linux example:

```bash
export MME_REALWORLD_RS_ROOT=/root/autodl-fs/datasets/MME-RealWorld-RS
python scripts/experiments/uhr_locator_param_sweep.py \
  --base-config configs/locator/uhr_hierarchical.yaml \
  --experiment-config configs/experiments/uhr_locator/diagnostic_v1.yaml \
  --manifest configs/experiments/uhr_locator/diagnostic_v1_samples.yaml \
  --output-dir artifacts/uhr_locator_sweeps/diagnostic_v1
```

Use repeated `--preset` and `--sample-id` flags for a small slice. `--resume` or
`--skip-existing` reuses successful `result.json` files after an interrupted
run. Suggested first commands isolate provider behavior:

```bash
# LAE only, tiled preset and one tiny-object sample
python scripts/experiments/uhr_locator_param_sweep.py ... \
  --preset T2_tile_1333_o15 --sample-id mme_rs_count_3422_airplanes \
  --disable-retriever

# VisRAG only on the same sample
python scripts/experiments/uhr_locator_param_sweep.py ... \
  --preset E0_baseline --sample-id mme_rs_count_3422_airplanes \
  --disable-detector --retriever-provider visrag

# Combined providers
python scripts/experiments/uhr_locator_param_sweep.py ... \
  --preset T2_tile_1333_o15 --sample-id mme_rs_count_3422_airplanes \
  --retriever-provider visrag
```

The ellipsis means the four common path arguments from the complete command
above. Review contact sheets and traces before running later parameter families;
do not promote averages from this diagnostic-only set to final hyperparameters.
