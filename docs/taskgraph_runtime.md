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
named entities is expanded by a configurable halo, cropped, and marked with stable
`A/B/C`, `REF`, `SUBJECT`, and `REFERENCE` labels. Canvas metadata retains the crop's
global origin, every global bbox, and the candidate-ID-to-entity mapping. Configure the
halo with `input_composer.candidate_halo_ratio` (default `0.2`).

Region retrieval treats a `Region` input as a hard search scope. Crop-local provider
boxes are mapped back to absolute original-image pixel coordinates and clipped to the
scope. `REGION_FROM_BBOX` uses the same coordinate convention and scales dataset boxes
when `params.image_size` differs from the loaded image.

## Operator mapping

| Operator | Capability | Current provider |
| --- | --- | --- |
| `REGION`, `REGION_FROM_BBOX` | geometry | deterministic Python |
| `FIND_MARKER` | artificial marker CV | deterministic Pillow implementation |
| `GROUP`, `ABS_DIFF` | typed deterministic transform | deterministic Python |
| `BUILD_ROUTE_CONTEXT` | route context construction | deterministic geometry/markers |
| `LOCATE` object target | object perception | **REAL:** existing LAE-DINO `ProposalProvider` adapter |
| `LOCATE` semantic region | region retrieval | existing UHR Locator or score-only Retriever adapter |
| `COUNT(image/Region)` | tiled detection/count | **REAL:** existing LAE-DINO; optional Retriever high-recall gate; global NMS |
| `COUNT(EntitySet)` | cardinality | deterministic Python; detector is not called again |
| `SELECT` rank/ordinal/extreme | geometry | deterministic Python |
| `SELECT` fuzzy relation | semantic selection | **REAL:** Qwen3-VL-2B |
| `ATTRIBUTE`, `CLASSIFY`, `MULTILABEL_CLASSIFY`, `MOTION` | visual semantics | **REAL:** Qwen3-VL-2B |
| `RELATION`, `VLM_REASON` | semantic reasoning | **REAL:** Qwen3-VL-2B |
| `ROUTE_REASON` | route semantics | **REAL:** Qwen3-VL-4B route role |
| final choice | visual/structured/mixed choice | **REAL:** Qwen3-VL-2B |

The LAE adapter consumes the existing dependency-light `ProposalProvider`, including
the isolated LAE sidecar and generic tiled wrapper. It converts crop-local boxes back
to absolute original-image coordinates and retains model/tile provenance. The Qwen
adapter lazily creates the existing `HuggingFaceVLMEngine` and reuses its processor,
model loading, device/dtype configuration, multi-image chat template, and generation
stack. No checkpoint is hard-coded and model files are never downloaded by this runtime.

## Provider status

REAL:

- LAE-DINO (`lae_dino_lae1m`, `lae_dino_dior`, or `lae_dino_dota`) through the
  existing sidecar registry; the `tiled` provider is the UHR count path.
- Qwen3-VL-2B for general semantic operations and choice resolution.
- Qwen3-VL-4B only for the route-specialist role.
- GeoRSCLIP (selected default), RemoteCLIP/FarSLIP, existing UHR Locator, and
  VisRAG scoring implementations are available through model-agnostic adapters.

PLACEHOLDER / replaceable contract:

- Final local Planner checkpoint: `PlannerProvider` plus `FixturePlannerProvider`.
- Future distilled region retriever checkpoint. `RegionRetrieverProvider` already
  has fake, UHR Locator, and score-based real adapters. This contract returns grid
  candidate boxes; it does not claim that CLIP natively produces bounding boxes.
- Evidence sufficiency: typed contract plus `FakeEvidenceSufficiencyProvider`; it is not
  forced into the critical path.

## Failure and trace contract

Each graph node produces a compact trace containing its refs, resolved runtime types,
provider, latency, output type/summary, fallback, or a typed error. Errors include node
id, operator, provider, input refs, exception, and retryability. Trace output never
contains image bytes, tensors, or full crops. The router allows one explicitly configured
same-capability fallback; otherwise failure is surfaced and is never silently converted
into a different semantic operation.

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
