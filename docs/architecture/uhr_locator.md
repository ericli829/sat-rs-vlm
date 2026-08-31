# Query-aware UHR Locator

The UHR Locator is a dependency-light, replaceable region-search layer for
ultra-high-resolution remote-sensing VQA. It selects one or more original-image
regions and produces an auditable trace; it does not answer the question itself.

## Architecture

```text
Question -> QueryParser -> TaskSpec -> TaskRouter -> SearchPlan
                                              |
                         HierarchicalLocator -+
                           |       |       |
        TiledProposalProvider  Retriever  deterministic spatial prior
              |                (question)        (geometry)
        ProposalProvider
        (object evidence)
                           \       |       /
                           CompositeScorer
                                  |
           depth-pool normalization + global adaptive beam
                                  |
                    Core+Halo refinement and stop policy
                                  |
                  RegionFusion -> Multi-ROI -> AnswerModel
```

Only the final dependency-light `MultiROIRequest` / `AnswerModel` protocol is
exposed to a future answer model. FO1 or another VLM can consume those crops
through an adapter, but no answer model is a Locator dependency and this stage
does not provide a production answer adapter.

## Public schemas

`TaskSpec` is defined in `sat_rs_vlm.semantics` and contains:

- `raw_question`, `operation`, `targets`, `attributes`, and `relations`;
- `spatial_scope`, `scope`, `multi_instance`, and optional absolute-pixel
  `given_bbox`;
- parser `confidence`, `parser_source`, and explicit `warnings`.

The first parser is rules-first and reuses
`configs/eval/semantic/remote_sensing_ontology.json`. Unresolved intent remains
`unknown`; it is not guessed. The old evaluation semantic imports are
compatibility wrappers around this common layer.

`SearchPlan` contains `use_detector`, `use_retrieval`, `use_spatial`,
`bypass_locator`, `desired_multi_region`, `target_phrases`, `route`, and
`warnings`. Routing is based on visual requirements, never dataset names.
Explicit bounding boxes and global-scene questions bypass hierarchical search.

`LocatorResult` returns absolute original-image `xyxy` regions, scores,
`TaskSpec`, `SearchPlan`, the complete search trace, three explicit area
metrics, maximum depth, latency, provider/model provenance, warnings, and final
region details. `processed_area_ratio` is cumulative scored-view work and may
exceed one. `selected_union_area_ratio` is the union of final ROIs and is always
in `[0, 1]`; `processed_union_area_ratio` is the union footprint of every scored
view. `inspected_area_ratio` remains a deprecated JSON/property alias for
`processed_area_ratio` only.

## Providers and scoring

`ProposalProvider` answers *what objects are where*. It returns global proposal
boxes and confidence. `DetectorRegionScorer` distributes each proposal across
every intersecting candidate using:

```text
coverage(B, R) = area(intersection(B, R)) / area(B)
score(R)       = sum(confidence(B_i) * coverage(B_i, R))
```

This preserves evidence for objects crossing grid boundaries. Existing LAE-DINO
and Grounding DINO providers are reused through their registry; the Locator does
not import either implementation.

For ultra-high-resolution images, a whole-image detector resize can erase tiny
objects before proposal generation. `TiledProposalProvider` is therefore a
generic wrapper around any proposal provider. It crops configurable overlapping
tiles, calls an unaware base provider, converts tile-local boxes back to absolute
original-image coordinates, and performs query-local global NMS. Its model/cache
identity includes tile size, overlap, NMS threshold, and top-K. Metadata retains
tile boundaries, local/global raw proposals, deduplicated proposals, kept raw
indices, and base/wrapper latency. Per-tile counts must never be added directly.

`RetrieverProvider` answers *which crop is relevant to the complete question*.
It must return one finite score per input region in the same order. The current
providers are deterministic `mock`, a lazy local-only `visrag` adapter, and a
generic lazy HuggingFace `clip` adapter. The latter is available through the
`clip`, `siglip`, `git_rsclip`, and `satelliteclip` aliases. Transformers-packaged
CLIP models use the HuggingFace backend. Official
GeoRSCLIP/RemoteCLIP/FarSLIP OpenCLIP checkpoints use the in-process OpenCLIP
backend in `integrations/retrievers/openclip.py`.
VisRAG encodes the question once per provider lifetime, batches original-image
crops, applies the official retrieval instruction and weighted-mean pooling with
L2 normalization, keeps raw cosine scores in metadata, performs no generation, and supports an
optional score cache whose key includes image hash, bounding box, query,
provider/model identity, and parameters. The checked-in profile uses a long-lived
JSONL sidecar because official VisRAG pins Transformers 4.40.2 while the shared
project environment uses Transformers 5.x. A direct runtime remains available
for an already compatible environment.

The OpenCLIP backend batches tile encoding and has four cache layers: bounded
decoded-image and CPU image-embedding LRUs, a provider-lifetime query embedding
cache, and an optional atomic disk score cache. Cache metadata is returned with
every score batch for benchmark auditing.

`SpatialRegionScorer` is model-free. It applies deterministic priors for image
parts such as left, north, center, or lower-right. It reports unavailable for a
complex object-object relation unless an anchor box is known.

For directional queries, the production GeoRSCLIP profile treats this prior as a
coarse first-pass hint: `spatial_first_depth_only: true` enables it only when
evaluating the initial depth-1 grid. The later refinement depths mark the spatial
component unavailable, so semantic retrieval can choose a target within the
selected branch without being pulled toward the original quadrant again. The
GeoRSCLIP profile uses `w_spatial=0.8` for that first pass (versus
`w_retrieval=1.0` and `w_parent=0.25`); active weights are renormalized, making
the directional prior about 39% of the first-pass composite weight. This is a
ranking prior, not a hard crop (`spatial_prefilter` remains false).

The composite score is configuration driven:

```text
w_detector * detector + w_retrieval * retrieval + w_spatial * spatial
+ w_parent * parent - w_redundancy * redundancy
```

Unavailable components are omitted and the active positive weights are
renormalized. Raw and depth-pool-normalized component values remain in each trace
entry.

## Core + Halo and hierarchical search

Each `SearchRegion` has a mostly non-overlapping logical `core_xyxy` and an
observation `view_xyxy`. The view expands the core by `halo_ratio` and clamps it
to image bounds. Every coordinate crossing a module boundary is an absolute
original-image pixel `xyxy`; crop-local and resized coordinates are not exposed.

At each depth, every current parent is divided by configurable `grid_size` (3 by
default). All children from all parents form one candidate pool and are scored
in one batch where possible. Component normalization and adaptive selection are
global within that depth, while `parent_id` and child lists preserve the tree.
Consequently `max_beam` caps the entire next frontier, not each parent.

Fused scores are standardized within the depth pool before softmax:

```text
z_i = (score_i - mean(score)) / max(std(score), epsilon)
p_i = softmax(z_i / temperature)
```

Equal scores yield uniform finite probabilities. The smallest selection reaching
`cumulative_mass` is retained, bounded by `1 <= K <= max_beam`; an overlap
penalty encourages diversity. Trace rows retain the raw fused score,
standardized logit, probability, effective logit, entropy, and selected K. Raw
scores from different depths are not treated as calibrated cross-level values.

Refinement stops only on target view size, maximum depth, maximum evaluated
regions, or cumulative processed-area budget. Posterior concentration controls
beam width rather than stopping a promising branch. Raw parent/child score gain
is intentionally excluded because independently normalized depths are not
comparable. These are diagnostic defaults, not tuned hyperparameters.

## Multi-ROI fusion

`RegionFusion` clamps and removes invalid regions, suppresses overlapping
regions, optionally merges adjacent regions, applies a final context margin, and
preserves global-coordinate provenance. Counting consumers must deduplicate
objects in this global coordinate system; counts from independent crops must not
be added directly.

## Adding providers

To add a detector, implement the existing
`integrations.detectors.protocol.ProposalProvider`, keep heavy imports lazy, and
register the constructor in `integrations.detectors.registry`. The Locator and
detector scorer require no changes.

To add a retriever, implement
`integrations.retrievers.protocol.RetrieverProvider`, validate output order and
length through `RetrievalResult`, keep model loading in the provider lifecycle,
and add a lazy registry branch. A future lightweight VisRAG student, SigLIP, or
distilled retriever therefore replaces only this provider. If its dependencies
conflict with the main environment, use a JSON sidecar following the existing
LAE-DINO pattern.

The offline comparison CLI accepts JSONL rows with `image`, `query`, and
absolute-pixel `gt_boxes` fields:

```powershell
python scripts/retriever_benchmark.py --manifest manifest.jsonl --provider siglip `
  --model-path C:\models\siglip --cache-dir .cache\siglip --output reports/siglip.json
```

Use the same grid, K, and thresholds for VisRAG, CLIP, SigLIP, and Git-RSCLIP.
Reports include Recall@K, GT coverage, selected area ratio, latency,
cache hits, and Count gate metrics (`gate_recall`, `detector_call_reduction`).
Use `scripts/analyze_retriever_gate.py` to calibrate a score threshold on an
image-disjoint calibration split before enabling a Count reject gate.

FO1 is intentionally optional: it is an answer/proposal experiment with a
separate compatibility surface, while localization must remain usable by any
answer model and in CPU-only tests.

## Configuration and commands

The checked-in configuration defaults to mock providers, so it is safe on CPU:

```powershell
python scripts/locator/run_uhr_locator.py `
  --config configs/locator/uhr_hierarchical.yaml `
  --image C:\path\image.png `
  --question "How many airplanes are visible in the northern part?" `
  --output locator.json
```

The selected production RS-CLIP profile is separate from the dependency-free
test profile:

```powershell
$env:GEORSCLIP_CHECKPOINT = 'C:\models\GeoRSCLIP-ViT-B-32.pt'
python scripts/locator/run_uhr_locator.py `
  --config configs/locator/uhr_hierarchical.georsclip.yaml `
  --image C:\path\image.png `
  --question "Where is the harbor?" `
  --output locator.json `
  --export-crops outputs\crops `
  --export-debug-overlay outputs\overlay.png
```

Provider selection is lazy. Only variables belonging to a selected provider are
expanded and validated.

LAE-only:

```powershell
$env:LAE_DINO_SOURCE_ROOT = 'C:\path\LAE-DINO'
$env:LAE_DINO_CONFIG_LAE1M = 'C:\path\lae_dino_lae1m.py'
$env:LAE_DINO_CHECKPOINT_LAE1M = 'C:\path\model.pth'
$env:LAE_DINO_BERT_ROOT = 'C:\path\bert-base-uncased'
$env:LAE_DINO_PYTHON = 'C:\path\lae-python.exe'
python scripts/locator/run_uhr_locator.py --config configs/locator/uhr_hierarchical.yaml --image C:\path\image.png --question "How many aircraft are visible?" --detector-provider lae_dino_lae1m --disable-retriever
```

For UHR tiny-object work, select the tiled wrapper through configuration:

```yaml
detector:
  enabled: true
  provider: tiled
  base_provider: lae_dino_lae1m
  tiled:
    tile_size: 1333
    overlap_ratio: 0.15
    global_nms_iou: 0.4
```

The values above are a diagnostic preset, not an optimum.

VisRAG-only:

```powershell
$env:VISRAG_MODEL_PATH = 'C:\path\VisRAG-Ret'
$env:VISRAG_PYTHON = 'C:\path\visrag-env\python.exe'
python scripts/locator/run_uhr_locator.py --config configs/locator/uhr_hierarchical.yaml --image C:\path\image.png --question "Which region contains an airport?" --disable-detector --retriever-provider visrag
```

Set an optional `repo_path` in `retriever.config` only when a local official
checkout must be added to `sys.path`. It is not required for a normal
`trust_remote_code` checkpoint.

LAE + VisRAG uses the same environment variables and both overrides:

```powershell
python scripts/locator/run_uhr_locator.py --config configs/locator/uhr_hierarchical.yaml --image C:\path\image.png --question "How many aircraft are visible in the north?" --detector-provider lae_dino_lae1m --retriever-provider visrag
```

Real pytest smoke is opt-in:

```powershell
$env:RUN_UHR_LOCATOR_REAL_SMOKE = '1'
$env:UHR_LOCATOR_SMOKE_IMAGE = 'C:\path\image.png'
python -m pytest -q -s tests/smoke/test_uhr_locator_real.py
```

The VisRAG case runs object, spatial/relation, and open-semantic queries while
reusing one loaded provider. With `-s`, it emits each candidate's bounding box,
detector/retrieval/spatial/fused scores, selection flag, latency, and provenance.

## Known limitations

- Rule parsing covers common deterministic forms but has no learned fallback.
- The VisRAG adapter follows the official remote-code representation API; a
  checkpoint/runtime-specific smoke is required before production use.
- Scorer weights, temperature, budgets, and thresholds have not been tuned on
  XLRS-Bench or MME-RealWorld-RS.
- Object-object relation anchoring and cross-crop counting deduplication are
  reserved for a future planner/consumer.
- `processed_area_ratio` measures cumulative crop work and may exceed 1.0;
  selected and processed union footprints remain bounded by one.
- The framework produces Multi-ROI crops and trace data, not final VQA answers.
