# VLM Semantic Execution

## Logical DAG and physical execution

The TaskGraph remains the logical description of **what** to compute. The runtime chooses
**how** an existing VLM operator executes. No fused operator is written into planner JSON.
`ExecutionPlan` analyzes final sources and downstream consumers, then supplies an immutable
per-node hint to `SemanticExecutor`.

Related frozen systems are documented in [Choice System](choice_system.md),
[SELECT System](select_system.md), and [TaskGraph Runtime](../taskgraph_runtime.md).

## Semantic decision layer

`SemanticDecisionLayer` is the output boundary for finite VLM semantics. It calls the
provider's generic `reason_and_decide` primitive. `LazyQwenSemanticProvider` implements
that primitive with the existing `HuggingFaceVLMEngine.reason_and_choose` cache machinery;
there is no second engine, cache implementation, M-RoPE implementation, or visual prefill.
The existing `ChoiceScoringRequest` API remains supported and adapts to the same primitive.

Free reasoning is retained only for trace/debug provenance. Canonical values are selected
from continuation scores and are never derived by regex, substring matching, comma splitting,
or parsing the reasoning prose.

## Intermediate finite semantics

The normal intermediate path is:

```text
operator -> InputComposer -> free reasoning -> same-model KV
         -> finite semantic scoring -> typed RuntimeObject -> RuntimeStore -> downstream
```

- `RELATION` scores the values of the canonical `SpatialRelation` enum and stores `Label`.
- `MOTION` scores `YES` against `NO` and stores `Boolean`. A score inside the configured
  uncertainty margin raises `SemanticDecisionUnresolvedError`; it never becomes `False` by
  parse failure.
- `CLASSIFY` with `label_space` scores one canonical label and stores `Label`.
- `MULTILABEL_CLASSIFY` reasons once, forks the shared cache for independent per-label
  `YES/NO` verification, and stores `LabelSet`.
- `ATTRIBUTE` uses cached categorical scoring only when configuration supplies an existing
  value space for that attribute. The runtime does not invent an attribute ontology.

Finite result provenance records provider, model ID, canonical scores and selections,
semantic method, cache reuse, latency, optional reasoning text, execution mode, and session
release metadata.

## Open semantics

`VLM_REASON` remains `Answer` during intermediate execution. `CLASSIFY` without a label
space and `ATTRIBUTE` without a configured value space remain free generation and store a
`Label` with `method=free_text_generation` and `canonical=false`. The schema currently
requires a finite label space for `MULTILABEL_CLASSIFY`, so there is no production free-text
split path for that operator.

## KV-cache lifecycle and safety

Every finite or fused decision uses the existing engine flow:

```text
reason_with_cache -> score_choice_from_cache -> finally session.close()
```

Qwen visual inputs are prefetched once. Candidate continuations reuse or fork only the text
KV state, according to decision mode. Cache objects, tensors, pixel values, and images never
enter a runtime object's provenance, `RuntimeStore`, or trace JSON. The provider exposes
trace-safe scalar metadata such as `visual_prefill_count` and `session_released`.

## Final-node choice fusion

The fused final path is:

```text
operator evidence + original question + original options
  -> free reasoning -> same-model KV -> benchmark option scoring
  -> ChoiceScoreResult -> RuntimeStore
  -> ChoiceResolver precomputed path -> ChoiceResult
```

This applies to eligible final-only `VLM_REASON`, `MATCH_CHOICE`, `ATTRIBUTE`, `CLASSIFY`,
`MULTILABEL_CLASSIFY`, `MOTION`, and `RELATION`. `ATTRIBUTE` and `RELATION` deliberately score
the original benchmark wording directly at the final boundary instead of first compressing
it through an intermediate ontology. Intermediate executions of the same logical operators
retain their normal typed output contracts.

`ROUTE_REASON` remains on its frozen same-4B path: 4B route reasoning and 4B cached option
scoring produce `ChoiceScoreResult`; a 2B model never consumes its cache.

## Fusion eligibility

Fusion is enabled only when all conditions hold:

1. final answer type is `CHOICE_SINGLE` or `CHOICE_MULTI`;
2. `final.sources` contains exactly one source;
3. the source operator is in the configured VLM allowlist;
4. no downstream node consumes that source;
5. original dataset options are non-empty; and
6. `final_vlm_choice_fusion.mode` is `auto`.

Fan-out, multiple final sources, free-form finals, non-allowlisted operators, missing options,
and `mode=off` disable the new fusion. The plan records an explicit `fusion_reason` and never
mutates the logical graph.

## Output contracts and transport

Runtime output validation is execution-dependent. For example, normal `ATTRIBUTE` must
produce `Label`, while an eligible final-fused `ATTRIBUTE` must produce `ChoiceScoreResult`.
The fused score object contains only IDs, scalar scores, reasoning text, provider/model
identity, timings, and trace-safe metadata. Because the node is final-only, no downstream
operator observes the physical override.

`ChoiceResolver` first preserves structured deterministic mapping for `ScalarInt`,
`ScalarFloat`, `Boolean`, `Label`, and `LabelSet`. When its sole source is a precomputed
`ChoiceScoreResult`, it validates the original option IDs and returns `ChoiceResult` without
another model call. `CHOICE_MULTI` retains zero, one, or many selections and always exposes
`choice_id=None`.

## Tracing

Each VLM node trace exposes `execution_mode`, `semantic_method`, `cache_reused`,
`final_choice_fusion`, and `fusion_reason`. Modes distinguish `intermediate_semantic`,
`final_choice_fused`, and `free_text`. Trace summaries truncate reasoning text and never
serialize visual or cache state.

## Failure semantics

Finite scoring failures are typed failures. Missing cached-decision capability, non-canonical
selected IDs, absent single selections, a non-reused cache, and binary uncertainty raise a
semantic decision or TaskGraph execution error. The runtime never substitutes `False`, a
first label, or an empty set because free reasoning text could not be parsed.

Configuration may declare finite attribute values only when backed by an existing ontology
or deployment contract. Open attributes remain intentionally non-canonical.
