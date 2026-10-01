# High-resolution TaskGraph Runtime

`sat_rs_vlm.taskgraph` is the production execution package for complex UHR
remote-sensing questions. `taskgraph_lab` remains an isolated planner training-data
laboratory and is not imported by production runtime code. The canonical production
contract is `sat_rs_vlm.taskgraph.schema.TaskGraph` (`taskgraph-v1.1`); a lab-exported
fixture is parsed by the compatibility test to catch contract drift.

## Dataflow

```text
Input
  |
  v
ExecutionModeRouter
  +-- DIRECT_VLM --------------------------> Qwen3-VL-2B
  +-- DIRECT_DETECTION / COUNT ------------> LAE-DINO
  |                                              |
  |                                              v
  |                                        typed result
  |
  +-- TASKGRAPH_UHR
          |
          v
     PlannerProvider
          |
          v
       TaskGraph
          |
          v
     GraphExecutor ---> CapabilityRouter ---> operator capability/provider
          |                                      |
          +------------------+-------------------+
                             v
                        RuntimeStore
                     $n1 -> Region
                     $n2 -> EntitySet
                     $n3 -> ScalarInt
                             |
                final.sources + final.question
                             |
                             v
                       InputComposer
                  visual | structured | mixed
                             |
                  ORIGINAL dataset options
                             |
                             v
                       ChoiceResolver
                             |
                             v
                         A/B/C/D/E
```

VRSBench caption/VQA and LEVIR-CC change understanding use `DIRECT_VLM` by
default. VRSBench count/detection uses `DIRECT_DETECTION`. MME RealWorld RS and
XLRS default to `TASKGRAPH_UHR`. The policy is explicit YAML, so dataset category
names can be aligned with actual adapters without training a router model.

## Typed execution

Every node reads and writes a runtime object: `ImageRef`, `Region`, `RegionSet`,
`Entity`, `EntitySet`, `ScalarInt`, `ScalarFloat`, `Boolean`, `Label`, `LabelSet`,
`RouteContext`, `Evidence`, `EvidenceSet`, or `Answer`. Named input roles are
preserved, and variable-length `VLM_REASON.evidence` lists are resolved item by item.
The schema rejects missing, unexpected, mutually exclusive, and illegal list inputs;
the executor validates resolved runtime types before capability fallback is considered.

`MATCH_CHOICE` is deprecated and retained only for loading older fixtures. New planner
exports end at `final.sources` and use the runtime's single final `ChoiceResolver` stage.

`InputComposer` materializes visual values as original-coordinate crops or marked
views and emits structured values in a typed format. It supports mixed and multi-image
inputs without concatenating arbitrary `str(object)` values. If all final sources are
authoritative structured results, it adds no image. Consequently, an LAE count cannot
be silently overridden by asking Qwen to count the original image again.

Fuzzy `SELECT` and `RELATION` inputs use a local candidate canvas: the union of the
named entities is expanded by a configurable halo, cropped, and marked with temporary
canvas labels `A/B/C`, `REF`, `SUBJECT`, and `REFERENCE`. Canvas metadata retains the
crop's global origin, every global bbox, the canvas-to-global transform, and each
upstream `candidate_id`. `A/B/C` are never persistent IDs: `SELECT` assigns or preserves
`Entity.provenance["candidate_id"]` at its boundary, then maps the model's letters back
to those entities. Configure the halo with `input_composer.candidate_halo_ratio`
(default `0.2`).

Region retrieval treats a `Region` input as a hard search scope. Crop-local provider
boxes are mapped back to absolute original-image pixel coordinates and clipped to the
scope. `REGION_FROM_BBOX` uses the same coordinate convention and scales dataset boxes
when `params.image_size` differs from the loaded image. It also accepts a nested `Region`,
keeps bbox coordinates absolute, clips to that scope, and rejects reversed or disjoint boxes.

`FIND_MARKER` performs color masking plus connected components, returns every accepted
component as an absolute `Region`, and filters pixel noise, implausibly large color fields,
and shape-aspect mismatches. `GROUP` is a real deterministic row/column/cluster grouping
operation and records stable group IDs and group boxes.

## Operator mapping

| Operator | Capability | Current provider |
| --- | --- | --- |
| `REGION`, `REGION_FROM_BBOX` | geometry | deterministic Python |
| `FIND_MARKER` | artificial marker CV | deterministic Pillow implementation |
| `GROUP`, `ABS_DIFF` | typed deterministic transform | deterministic Python |
| `BUILD_ROUTE_CONTEXT` | route context construction | deterministic geometry/markers |
| `LOCATE` object target | object perception | **REAL:** existing LAE-DINO `ProposalProvider` adapter |
| `LOCATE` semantic region | region retrieval | existing UHR Locator or score-only Retriever adapter |
| `COUNT(image/Region)` | tiled detection/count | **REAL:** existing LAE-DINO; existing tiled wrapper handles global transform/NMS |
| `COUNT(EntitySet)` | cardinality | deterministic Python; detector is not called again |
| `SELECT` rank/ordinal/extreme | geometry | deterministic Python |
| `SELECT` `LEFT_OF`/`RIGHT_OF`/`ABOVE`/`BELOW`/`INSIDE`/`OVERLAP` | geometry first | deterministic Python; boundary cases fall back to Qwen3-VL-2B |
| `SELECT` `NEAR`/`NEXT_TO`/`AROUND`/`BETWEEN` | semantic selection | **REAL:** Qwen3-VL-2B |
| `ATTRIBUTE`, `CLASSIFY`, `MULTILABEL_CLASSIFY`, `MOTION` | visual semantics | **REAL:** Qwen3-VL-2B |
| `RELATION`, `VLM_REASON` | semantic reasoning | **REAL:** Qwen3-VL-2B |
| `ROUTE_REASON` | route semantics | **REAL:** Qwen3-VL-4B route role |
| final choice | deterministic mapping or same-model KV-cached constrained choice | **REAL:** shared 2B, or the active 4B Route session |

Route V1 resolves singleton endpoints or a unique highest score only when every candidate
is scored; ties and partially scored sets are explicit ambiguity errors. Context geometry
uses an endpoint union, margin, minimum side, image clipping, optional aspect-preserving
resize, and runtime START/GOAL overlays. The route prompt treats endpoint descriptions as
already-resolved identity context while preserving navigation constraints. Trace-safe score
metadata includes prompt version, crop/render sizes, global-to-render transform, marker style,
provider/model identity, cache reuse, and latency. Route reasoning and option scoring stay in
one 4B cached session; no route-specific answer parser exists.

The LAE adapter consumes the existing dependency-light `ProposalProvider`, including
the isolated LAE sidecar and generic tiled wrapper. It converts crop-local boxes back
to absolute original-image coordinates and retains model/tile provenance. The Qwen
adapter lazily creates the existing `HuggingFaceVLMEngine` and reuses its processor,
model loading, device/dtype configuration, multi-image chat template, and generation
stack. No checkpoint is hard-coded and model files are never downloaded by this runtime.

## SELECT v1 contract

`SELECT` is a TaskGraph operator, not a retriever and not a new top-level task type.
It only filters candidates supplied by an upstream node such as `LOCATE` or an existing
Region Retriever. It never imports, invokes, or recursively searches a retriever.

Inputs are `candidates: EntitySet | Region | RegionSet`, optional `reference`, and an
optional current `scope: ImageRef | Region`. All coordinates are absolute original-image
pixels in `bbox_xyxy_global`; normalized coordinates are not accepted. A reference used
for deterministic geometry must resolve to exactly one entity or region. A plural
reference is retained only as visual context for semantic fallback.

Every execution returns `SelectResult`, containing `selected`, a status (`OK`, `EMPTY`,
`AMBIGUOUS`, `UNRESOLVED`, or `ERROR`), method (`geometry`, `qwen3_vl_kv_cached_choice`,
or a clearly marked compatibility fallback), optional confidence, and trace provenance.
The capability router applies one shared unwrap policy. `COUNT`, `GROUP`, and a second
`SELECT` explicitly accept empty sets; single-object consumers such as `ATTRIBUTE` and
`CLASSIFY` unwrap a singleton set to its entity or region. `AMBIGUOUS`, `UNRESOLVED`,
and `ERROR` are rejected before another operator, `InputComposer`, or `final.sources`
can materialize them. A multi-item result is also rejected by a single-object consumer.

All candidates, reference objects, and the current scope must resolve to one source image.
Mixed-image inputs return `UNRESOLVED` with `cross_image_select_inputs`; coordinates from
different images are never compared. The default relation margin is `max(4 px, 2% of the
smaller current-scope dimension)`. For a Region scope, its bbox width and height are used,
not the dimensions of the full UHR image.

`SUBREGION` constructs a global rectangle from the current search scope and the reference
box: `LEFT_SIDE`, `RIGHT_SIDE`, `ABOVE`, `BELOW`, `INSIDE`, and `AROUND` are supported.
`OUTSIDE` and `BOTH_SIDES` intentionally return `UNRESOLVED` in v1; polygon ROI,
arbitrary complements, multi-hop selection, generic learned ranking, and SELECT-to-
retriever calls are outside this operator's scope.

Relations are partitioned into deterministic positive, grey, and negative candidates.
Only grey candidates appear on the Qwen candidate canvas. Clear positives are retained,
clear negatives are never re-evaluated, and canvas labels are mapped back to stable
`candidate_id` values. Qwen3-VL first performs free visual reasoning and then reuses the
same Transformer KV cache for independent per-candidate YES/NO verification. Both
`selection_type=SINGLE` and `MULTI` use this `CHOICE_MULTI`-style verification primitive;
SELECT applies its own cardinality policy afterward: SINGLE maps 0/1/N matches to
EMPTY/OK/AMBIGUOUS, while MULTI maps 0/N to EMPTY/OK.

This is intentionally different from final benchmark `CHOICE_SINGLE`. A benchmark single
choice guarantees one legal answer and uses argmax. SELECT SINGLE does not guarantee that
any candidate satisfies the predicate and therefore must be able to return EMPTY or
AMBIGUOUS instead of forcing one object.

Finite-set token-masked decoding remains only as a compatibility fallback when the backend
explicitly raises `CachedChoiceUnavailableError`. Runtime, CUDA, mapping, Transformers, and
other unexpected failures propagate. Given at most eight candidate labels, the fallback
accepts only canonical finite outputs (`NONE`, `A`, `B`, `A,C`, and so on) and records its
reason and type. With more than eight candidates, or without confirmed constrained
decoding, SELECT returns `UNRESOLVED` with `safe_fallback_unavailable`; unrestricted prose
generation plus regex parsing is never a production fallback.

RANK accepts only the serialized criteria `bbox_area` and `score`; missing scores produce
`UNRESOLVED/rank_score_missing`. ORDINAL and EXTREME ties compare only their primary axis.
The full frozen design and provenance fields are documented in
[SELECT System](architecture/select_system.md).

Choice details are frozen in [Choice System](architecture/choice_system.md). The default
2B choice capability aliases `semantic_2b`, fuzzy SELECT reuses that model's reasoning KV
cache, and Route reasoning plus final option scoring stays entirely inside one 4B session.
Free reasoning is retained for traceability but never regex-parsed into the final answer.
Choice results and trace summaries always include `answer_type`, `selected_ids`, and the
legacy scalar `choice_id`. The scalar is non-null only for `CHOICE_SINGLE`; a
`CHOICE_MULTI` result with exactly one selected option still reports `choice_id: null`.

## Provider status

REAL:

- LAE-DINO (`lae_dino_lae1m`, `lae_dino_dior`, or `lae_dino_dota`) through the
  existing sidecar registry; the `tiled` provider is the UHR count path.
- Qwen3-VL-2B for general semantic operations and choice resolution.
- Qwen3-VL-4B only for the route-specialist role.
- Existing UHR Locator and VisRAG scoring implementations are available through adapters.

Replaceable contract:

- Final local Planner checkpoint: `PlannerProvider` plus `FixturePlannerProvider`.
- Lightweight region retriever/checkpoint: `RegionRetrieverProvider` plus fake,
  UHR Locator, and score-based adapters. This contract returns candidate boxes; it does
  not claim that CLIP natively produces bounding boxes.
- Evidence sufficiency: `EvidenceSufficiencyExecutor` reuses `semantic_2b` and its existing
  finite KV-cache primitive. It returns only `SUFFICIENT`, `NEED_MORE_EVIDENCE`,
  `UNRESOLVED`, or `ERROR` plus trace-safe confidence/metadata; reasoning text is never
  exposed. It is an auxiliary runtime service, not a TaskGraph operator, and never controls
  exhaustive counting or graph traversal. See [Answerability](architecture/answerability.md).

## Failure and trace contract

Each graph node produces a compact trace containing its refs, resolved runtime types,
provider, latency, output type/summary, fallback, or a typed error. Errors include node
id, operator, provider, input refs, exception, and retryability. Trace output never
contains image bytes, tensors, or full crops. The router allows one explicitly configured
same-capability fallback; otherwise failure is surfaced and is never silently converted
into a different semantic operation.

The batch evaluator additionally writes a sample-level `reasoning_chain`. It records the
planner's generated DSL and validation attempts, the canonical TaskGraph, every module's
inputs/provider/output/provenance, final source references, model-generated intermediate
text, and all persisted visual artifact paths. If a sample fails after graph execution,
the partial node trace is retained with the failure stage and message. The evaluator also
writes `answer_judgment` with normalized choice/text comparison when the input contains
`ground_truth`, `Ground truth`, `reference_answer`, or `answer`; these reference fields
are used only after runtime inference and are never passed to a model.

## CLI

Fake single-sample execution:

```powershell
python -m sat_rs_vlm.taskgraph.run `
  --image tests/fixtures/miniature_dataset/images/counting.ppm `
  --question "How many ships are there?" `
  --dataset MME_RealWorld_RS `
  --task count `
  --sample-id count-demo `
  --question-type MULTIPLE_CHOICE_SINGLE `
  --options-json '["A 5","B 6","C 7","D 8"]' `
  --provider-config configs/taskgraph/runtime.fake.yaml `
  --trace-output outputs/taskgraph/count-demo.trace.json
```

The same command is available as `sat-rs-vlm taskgraph run`. Pass a full graph with
`--graph-json path/to/taskgraph.json`, or configure a fixture planner. Real providers
require `--real-model` and paths supplied through environment-backed config.

Batch evaluation with `scripts/taskgraph/evaluate_runtime.py` writes each row's
`answer`, `answer_judgment`, resolved `input_image_paths`, persistent
`intermediate_output_paths`, and an explainability-oriented `reasoning_chain`.
When `--artifact-dir` is omitted, generated visual inputs are stored beside the
JSONL output in `<output-stem>_artifacts`; pass an explicit directory when the
artifacts must be retained across runs or devices.

## Verification

CPU/offline fake suite:

```powershell
python -m pytest tests/unit/taskgraph/test_runtime.py -q
```

Opt-in real smoke tests:

```powershell
$env:TASKGRAPH_REAL_CONFIG = "configs/taskgraph/runtime.real.example.yaml"
$env:TASKGRAPH_SMOKE_IMAGE = "path/to/local/image.jpg"
$env:RUN_REAL_LAE = "1"
python -m pytest tests/smoke/test_taskgraph_real.py -q -k lae

$env:RUN_REAL_QWEN = "1"
python -m pytest tests/smoke/test_taskgraph_real.py -q -k qwen
```

The required model/source environment variables are documented directly in the real
example config. Ordinary pytest runs skip these tests and require no GPU, model, or network.

## CPU image preprocessing

Region score caches retain their existing keys, but compute the source SHA-256 once
per scoring batch. GeoRSCLIP checks score and embedding caches before decoding pixels;
only missing embeddings need crops, which are materialized per inference batch.
Trace metadata includes `image_decode_skipped` and `actual_crop_count` alongside the
existing candidate `crop_count`.

Tiled detection prepares crops on CPU threads while the existing workers run
synchronous inference, preserving tile order and global coordinates. Each worker
prepares at most one tile ahead; COUNT keeps inference serial and also prepares
one tile ahead. Model instances, inference worker counts, batch sizes, and CUDA
inputs are unchanged. Set `tile_prefetch: false` on the tiled provider or COUNT
detector configuration to disable this overlap.
The default transport is lossless RGB BMP, which avoids PNG compression work.
Set `tile_image_format: png` on the tiled provider or counting detector configuration
when smaller transport files are needed. BMP uses more disk bandwidth; at most two
transport files per active tiled worker (active plus lookahead) are retained,
or one with prefetch disabled. Files are removed after prediction. A 1333×1333 RGB
BMP is about 5.1 MiB: five workers retain at most about 51 MiB with prefetch.
The decoded source is released after creating the full RGB image, before tiled
inference starts. The COUNT bridge exposes `call_count` and does not retain request
images. Temporary files use the system temporary directory.

Compare a committed baseline with the working tree using local UHR fixtures:

```powershell
python scripts/experiments/benchmark_image_preprocessing.py `
  --image-root D:/Desktop/tzb-2026/results/uhr_locator `
  --baseline-ref c77e098 --repeats 2 `
  --output .tmp/image-preprocessing-20261001/summary.json
```

The benchmark runs the committed code and current code on identical images in
alternating order. It checks score-cache results, transported tile pixels, global
boxes, and proposal order. A deterministic pixel reader replaces model inference;
reported speedups apply to CPU image handling, not complete-system latency.
Timing includes source PNG/JPG decoding, RGB conversion, cropping, transport-file
writing, and transport-file reading. Optional `--inference-delay-ms` adds a
simulated wait to each inference call to measure overlap; it is explicitly not a
real-model benchmark. `--workers 1` evaluates overlap without parallel inference.

BMP transport is compatible with the pinned LAE-DINO source revision
`6b1519626e39d1f39f8ed1f38761c20f7e0e8c35`. Its three checkpoint configurations
use `LoadImageFromFile` with MMCV's default OpenCV decoder, `FixScaleResize`,
`PackDetInputs`, and `DetDataPreprocessor(bgr_to_rgb=True)`. The transport writes
standard uncompressed 24-bit BMP. The decoder handles BMP's bottom-up rows and
produces BGR, followed by the existing BGR-to-RGB conversion and normalization.
No additional channel swap or vertical flip belongs in the transport code.
[OpenCV documents BMP support](https://docs.opencv.org/4.9.0/d4/da8/group__imgcodecs.html);
[MMCV documents its image decoder](https://mmcv.readthedocs.io/en/2.x/api/generated/mmcv.image.imfrombytes.html).
Grounding DINO reads the same files through Pillow and converts them to RGB.

Check actual model inputs without allocating another model or CUDA tensors:

```powershell
python scripts/integrations/check_detector_image_transport.py `
  --image-root D:/Desktop/tzb-2026/results/uhr_locator `
  --lae-source-root "$env:LAE_DINO_SOURCE_ROOT" `
  --output .tmp/detector-image-transport/summary.json
```

The checker requires CPU Torch, Pillow, NumPy, OpenCV, MMCV (mmcv-lite suffices),
and MMEngine. Optional `--dependency-root` loads an isolated dependency directory.
It reads the selected LAE config settings and executes their image-processing
definitions, omitting registry decorators and annotation-only box casting to
avoid importing unrelated CUDA operators. It does not change runtime configs.
On the three local UHR fixtures, 36 PNG/BMP comparisons across LAE-1M, DIOR, and
DOTA passed: decoded BGR pixels and final float32 NCHW CPU tensors matched exactly
(maximum absolute difference 0). Cases include first/middle/last tiles, width 7
to test BMP row padding, asymmetric rows, grayscale, and RGBA conversion. This
checks decoding and model input preprocessing; it does not run checkpoint forward
inference or measure GPU output differences.
