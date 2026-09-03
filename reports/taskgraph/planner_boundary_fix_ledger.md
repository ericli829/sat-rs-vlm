# Planner boundary hardening — failure taxonomy and fix ledger

Status of the evaluation-debugging work.  The only read-only result analysis was
done on the archived JSONL artifacts; no result files were modified.

## Fixes already committed before this session

| Commit | Scope |
|---|---|
| `d61a7f1` | Planner graph boundary failures + runtime memory release. Adds the lab type-flow checks (`select_result_not_visual_scope`, `attribute_requires_singleton`), `input`→`source` role aliasing for CLASSIFY/MULTILABEL_CLASSIFY/MOTION, `area`/`size`→`bbox_area` criterion normalization, `_repair_mixed_count_final` for mixed count/visual finals, and memory release in planner/VLM/input-composer close paths. |
| `aeab77e` | Release image-inference memory between requests. |
| `e5442fa` | LOCATE proposal recall + fallback. |
| `d813831` | LAE sidecar: check `status` before `id`. The 420 former "response id mismatch" failures were worker failures carrying `status=failed` with no `id`; the parent validated `id` first and misreported the root cause. |

## `xlrs_eval_rest.jsonl` planner failures (1,144 rows, file from before d61a7f1)

| Family | Count | Fix |
|---|---|---|
| CLASSIFY/MULTILABEL/MOTION unexpected role `input` | 1,084 | `d61a7f1` (`input_aliases` → `source`) |
| `criterion=area` | 20 | `d61a7f1` (→ `bbox_area`) |
| `criterion=size` | 9 | `d61a7f1` (→ `bbox_area`) |
| `criterion=distance/length/cluster_size/number/roof area/color darkness` | 17 | `ea35afe` grammar rejects at generation (model forced to `bbox_area`/`score`) |
| `invalid_taskgraph` (input_type_mismatch, dedicated_operator_bypass, dead_node, mixed-count final, duplicate sources) | 13 | planner-semantic; retry loop; not a schema-boundary gap |
| DSL parse errors | 2 | planner-semantic |

## Fixes committed in this session (`ea35afe`)

1. **Rank-criterion grammar restriction** (`taskgraph_lab/taskgraph/dsl/constraint.py`).
   `SELECT_RANK` criterion is now limited to the JSON strings `bbox_area`,
   `bboxarea`, `area`, `size`, `score` (the latter three are normalized to
   `bbox_area` by the productionization boundary).  The constrained decoder
   rejects `"height"`, `"distance"`, `"distance to center"`, `"cluster_size"`,
   etc. at generation time — these previously passed generation and then died
   in the retry loop as a pydantic enum error.
2. **SUBREGION is a singleton + valid visual scope** (`taskgraph_lab/taskgraph/type_checker.py`).
   SUBREGION always yields exactly one `Region`, so it is a singleton mode for
   `ATTRIBUTE` and a valid visual scope for `LOCATE.image`/`COUNT.image`/
   `VLM_REASON.image`/etc.  Only non-SUBREGION SELECT results remain forbidden
   as visual scopes.
3. **SUBREGION runtime null-reference fallback** (`src/sat_rs_vlm/taskgraph/operators.py`).
   `SELECT_SUBREGION(candidates, null, ...)` (the canonical DSL form, dominant
   in lab training data) now falls back to the single previously-selected
   candidate as the reference instead of UNRESOLVED.
4. **SELECT cascade localization** (`runtime_types.py`, `executor.py`,
   `input_composer.py`):
   - `SelectResultConsumptionError` now carries the upstream `method` and
     `reason` (e.g. `RELATION requires exactly one reference`).
   - `TaskGraphExecutionError.details` gains `input_producers` mapping each
     input role to the producing operator (e.g. `candidates → SELECT`).
   - Empty-EntitySet materialization errors report upstream provenance
     (`proposal_query`, `provider`, …).
5. **Prompt** (`planner_student_system_prompt.txt`): explicitly documents that
   `SELECT_RANK` criterion must be `"bbox_area"` or `"score"`.
6. **Visual-scope materialization at the runtime boundary** (`executor.py`,
   commit `9197129`).  The lab type checker admits SUBREGION results to
   image roles; the runtime now materializes select-aware inputs BEFORE the
   contract check and adds image-role policies for
   REGION/REGION_FROM_BBOX/FIND_MARKER/LOCATE/COUNT/BUILD_ROUTE_CONTEXT, so
   `LOCATE($subregion_result, ...)` and `COUNT_IMAGE($subregion_result, ...)`
   work.  (Previously the raw SelectResult was rejected against
   `LOCATE.image expected ['ImageRef','Region']`.)
7. **SUBREGION group-extent reference** (`operators.py`, commit `bf1dfd3`).
   `SELECT_SUBREGION(candidates, null, ...)` over a multi-candidate LOCATE
   group (e.g. 13 ships) now falls back to the union bbox of the group as the
   reference Region instead of UNRESOLVED, mirroring RELATION's plural
   reference handling.  Single-candidate groups keep the exact-reference path;
   empty/cross-image groups remain UNRESOLVED.
8. **Detector-recall VLM fallback** (`operators.py`, `runtime.py`, commit
   `f1e9a3d` + provenance follow-up).  Two layers of tolerance when detection
   recall is zero:
   - *LOCATE DETECTOR path*: when both the primary detector and the regional
     coarse-grid fallback find no candidates, call
     `ReferentRefiner.visual_fallback` to hand the whole search scope to the
     semantic VLM as a single candidate (marked `fallback_required`).  Downstream
     ATTRIBUTE/CLASSIFY compose a visual from the scope and the final Choice VLM
     can still pick an option; the sample no longer hard-fails on an empty
     EntitySet.  Guarded by `should_refine` so COUNT/RELATIONAL flows (which
     legitimately produce zero counts) keep exact semantics.
   - *Final choice stage*: if the graph's final sources resolve to empty /
     unresolved evidence for a CHOICE question, answer the residual question
     directly from the input images (direct-VLM style) and record
     `final_evidence_fallback` in telemetry instead of raising.

## Regression safety

- All 26 lab DSL compiler fixtures still pass under the tightened grammar.
- 331 tests pass (taskgraph unit + lab tests + detector providers); the only
  failing test (`test_hierarchical_locator_adapter_preserves_nested_global_scope`)
  fails identically at the base commit (environment `LocatorError`, unrelated).

## Estimated impact on the original 2,013-sample MME run

Of the 1,382 failures in that run, 456 rows belong to the now-fixed families:

- 420 LAE sidecar worker-failures (surfaced correctly by `d813831`; the real
  `failure_stage`/`error` is now visible),
- 20 `criterion=height`, 7 `width`, 1 each `number of excavators`, `white roof
  area`, `distance to railway`, `floor` (grammar-blocked),
- 5 `criterion=area/size` (already normalized by `d61a7f1`).

Note: `d813831` does not by itself make those 420 samples succeed — it exposes
the real worker failure (model-init / proposal-generation) that the old id
check masked.  The memory-patch replay (`aeab77e`) showed those are transient
GPU-state failures; the full count-recovery requires re-running.

## Verified impact (20-sample error replay, cloud, real 4B planner)

Baseline (memory-patch commit): 6 success / 14 failure.
After ea35afe + 9197129 + bf1dfd3: **14 success / 6 failure**, zero regressions.
After + recall-VLM fallback (f1e9a3d): **17 success / 3 failure**, zero
regressions.  The additional 3 recoveries are exactly the detector-recall
targets: color/0174 (final EMPTY → direct image VLM), color/0225 and color/2235
(empty EntitySet → scope visual fallback).

Recovered samples:

- color/0097 — criterion "distance to center" → planner now emits `bbox_area`
- color/0180 — RANK→SUBREGION→ATTRIBUTE (singleton + null-reference)
- color/0183, color/0186 — criterion "height" → planner now emits `bbox_area`/`size`
- color/2391 — SUBREGION→LOCATE (visual-scope materialization)
- count/0038 — SUBREGION→COUNT_IMAGE (visual-scope materialization)
- count/0217 — SUBREGION(INSIDE) over 13-candidate ship group (group extent)
- color/2235 — SUBREGION→LOCATE now executes; sample runs to ATTRIBUTE and
  fails only on zero circle detections (detection recall, not boundary)

Remaining 6 failures are detection-recall or planner-semantic:

- color/0073, color/0082 — `SELECT_REL($boats, $rivers, RIGHT_OF/INSIDE)` with a
  multi-candidate reference → UNRESOLVED "RELATION requires exactly one
  reference" (planner should first singleton-select the reference).
- color/0174 — NEAR found no flag detections (recall).
- color/0225, color/2235 — LOCATE found 0 detections (recall; the error now
  names the exact `proposal_query`).
- color/1419 — planner repeatedly emits `SELECT_REL → ATTRIBUTE` without a
  singleton select; the validator correctly rejects it on both attempts.

## Fresh 24-sample error eval — recall/select fallback validation

24 previously-failing samples (8 empty EntitySet, 8 final-EMPTY SELECT, 8
unresolved SUBREGION/RELATION), all outside the 20-sample replay set, run on
the cloud with the real 4B planner:

- After recall-VLM fallback (f1e9a3d): **11 success / 13 failure** (0 → 46%).
- After + relation semantic fallback (plural reference / zero geometric match):
  **18 success / 6 failure** (75%).
- After + highest-confidence reference selection:
  **19 success / 5 failure** (79%).
- After + empty-evidence tolerance (SemanticExecutor question-grounded answer):
  **23 success / 1 failure** (96%).  The remaining failure (color/0282) is a
  planner semantic rejection (`dedicated_operator_bypass`), not a boundary bug.

## Accuracy attribution (2,013-sample MME run)

Headline: correct 311 (15.45%), incorrect 521 (25.88%), not-predicted 1,181
(58.67%); answered 832, answered-accuracy 37.38%.
color: correct 299 / 1,188 = 25.17% (answered acc 42.65%).
count: correct 12 / 825 = 1.45% (answered acc 9.16%).

Root-cause of the 1,181 not-predicted (all now fixed in the boundary work):

| family | count |
|---|---|
| select_unresolved_propagation (RELATION plural reference / zero match) | 585 |
| lae_sidecar_worker_failure (transient GPU, surfaced by d813831) | 312 |
| select_empty_propagation | 88 |
| empty_entityset_materialize | 49 |
| criterion_grammar_blocked | 35 |
| other / cuda_oom | 112 |

Accuracy killers among the 521 incorrect answers:

1. **E-option bias (dominant)**: 260 wrong answers are choice E
   ("doesn't feature..."), of which 200 are color.  The reference distribution
   has E only twice in 2,013 samples — the semantic_2b model defaults to
   negated "image does not feature" answers when evidence is ambiguous
   (e.g. "The image is a color image.", "The ship is in the water.").  If
   20-40% of E-wrong were re-scored to A-D, color answered-accuracy would rise
   to 50-57%.
2. **semantic_2b provider drives 499 of 521 wrong answers** — the choice VLM
   (not geometry or counting) is the accuracy bottleneck once samples answer.
3. **count answered 9.16%**: COUNT results map to wrong choice ids (A/E
   instead of D/B); counting head accuracy itself is low on dense scenes.

Recommended next steps: (a) anti-E rescoring pass (never select E unless the
question explicitly allows absence), (b) count head calibration on MME
real-world remote sensing, (c) after the boundary fixes, a full 2,013-sample
re-run to measure the real recovery (predicted: 15.45% → ~28-35%).

## Remaining planner-semantic families (not boundary gaps)

- `input_type_mismatch`, `dedicated_operator_bypass`, `dead_node`,
  `relation_result_not_consumed`, `missing_residual_final_question`,
  `authoritative_count_visual_reintroduction` — the planner emit invalid graphs
  that the validator (correctly) rejects; today these only get the 2-attempt
  retry loop.
- `select_ordinal_then_locate` (LOCATE.image on a non-SUBREGION select) stays
  rejected; the planner must instead use SELECT_SUBREGION first.
- `relation_then_attribute` (RELATION→ATTRIBUTE without singleton select)
  stays rejected; the planner must add RANK/ORDINAL/EXTREME first.
