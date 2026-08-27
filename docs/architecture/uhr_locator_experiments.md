# UHR Locator experiment plan

This file records planned ablations. The checked-in defaults are starting points;
they are not benchmark-tuned values. Development sweeps must not use held-out
XLRS-Bench or MME-RealWorld-RS test labels.

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
- **Search Area Ratio:** cumulative inspected crop area divided by original image
  area. It may exceed one for multilevel/halo inspection.
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
