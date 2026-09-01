# SELECT System

## Boundary and output

`SELECT` filters candidates that already exist. It does not call a detector or retriever,
perform counting, plan another graph, or implement final benchmark choice. Inputs are an
`EntitySet`, `Region`, or `RegionSet`, an optional reference, and an optional current
`ImageRef` or `Region` scope. Every bbox is absolute original-image pixel XYXY in
`Region.bbox_xyxy_global`.

Every invocation returns a `SelectResult`:

- `OK`: the selected value is safe for an explicitly compatible downstream role.
- `EMPTY`: no candidate matched; only set-aware consumers that opt into empty semantics
  may unwrap it.
- `AMBIGUOUS`: more than one object remains where one was required, or a deterministic
  rank/axis tie cannot be resolved.
- `UNRESOLVED`: the requested selection cannot be decided safely.
- `ERROR`: the SELECT boundary or internal mapping is invalid.

The shared `unwrap_select_result` policy is applied by the capability router,
`InputComposer`, and final-source materialization. `COUNT`, `GROUP`, and SELECT candidates
accept an EMPTY set. `ATTRIBUTE`, `CLASSIFY`, `MULTILABEL_CLASSIFY`, `MOTION`, and semantic
RELATION roles require one object and unwrap a singleton `EntitySet`/`RegionSet` to its
item. `VLM_REASON` may consume an OK set as evidence. AMBIGUOUS, UNRESOLVED, and ERROR are
never sent implicitly to a VLM. Final sources accept only OK.

## Cardinality

`selection_type=SINGLE` is a result constraint, not a promise that a match exists:

| Verified match count | SINGLE | MULTI |
| --- | --- | --- |
| 0 | `EMPTY` | `EMPTY` |
| 1 | `OK` | `OK` |
| 2 or more | `AMBIGUOUS` | `OK` |

Final benchmark `CHOICE_SINGLE` has a different contract: the benchmark guarantees one
correct option, so argmax is appropriate. SELECT SINGLE can legitimately have zero or
multiple predicate matches. Semantic SELECT therefore always requests `CHOICE_MULTI`
independent YES/NO verification and applies the table above itself.

## Geometry-first relations

For `LEFT_OF`, `RIGHT_OF`, `ABOVE`, and `BELOW`, center coordinates and the configured
margin partition candidates into clear positive, grey, and clear negative sets. `INSIDE`
uses full containment/intersection/outside; `OVERLAP` uses above-threshold IoU,
below-threshold positive IoU, and zero IoU. Fuzzy relations such as `NEAR`, `NEXT_TO`,
`AROUND`, and `BETWEEN` place every candidate in grey.

Only grey candidates are drawn on the local candidate canvas and sent to Qwen3-VL-2B.
Clear positives remain selected; clear negatives are never re-evaluated. If SINGLE already
has two clear positives, SELECT returns AMBIGUOUS without a model call. One clear positive
plus grey candidates still requires verification because another positive would make the
result ambiguous.

The grey canvas may relabel its subset as A/B/C, but every canvas entry retains the
upstream stable `candidate_id`. Results map labels back through that ID, never through a
position in the full candidate tuple. Provenance records:

- `all_candidate_ids`
- `clear_positive_candidate_ids`
- `grey_candidate_ids`
- `clear_negative_candidate_ids`
- `semantic_positive_candidate_ids`
- `final_candidate_ids`
- provider, scores, cache reuse, latency, and compatibility fallback details

No cache tensors, pixel tensors, PIL objects, or image bytes are stored in provenance.

## Image and scope rules

Candidates, reference, and scope must all resolve to the same image path. A candidate set
or RegionSet containing multiple images, or any cross-image role combination, returns
`UNRESOLVED` with reason `cross_image_select_inputs`. Geometry never compares XYXY values
from different images.

An explicit margin wins. Otherwise SELECT uses:

```text
max(4.0, min(scope_width, scope_height) * 0.02)
```

For an `ImageRef`, dimensions are the image dimensions. For a Region scope, width and
height come from its bbox, so an 800x800 region in a 10000x10000 image uses a 16-pixel
default margin rather than 200 pixels.

## Cached semantic verification and safe fallback

The normal semantic path performs free reasoning once and reuses the same Qwen session KV
cache for per-candidate YES/NO scoring via `multi_verify_template`. Reasoning text is kept
only for traceability and is never regex-parsed to decide the selection.

Compatibility fallback is entered only for `CachedChoiceUnavailableError`, which means the
backend explicitly lacks cached choice capability. Up to eight candidates may use confirmed
finite-set token-masked decoding. The trace method becomes
`qwen3_vl_token_mask_fallback` and records `fallback_reason` and `fallback_type`. If the
candidate set is larger, or the provider does not confirm constrained decoding, SELECT
returns `UNRESOLVED/safe_fallback_unavailable`. CUDA OOM, invalid mappings, M-RoPE errors,
and other unexpected exceptions propagate instead of being disguised as compatibility.

`parse_selection_indices` remains for constrained-output and legacy unit compatibility;
it is not a production parser for unrestricted free prose.

## Deterministic modes

- RANK criteria are the finite serialized enum `bbox_area | score`. An unknown criterion
  fails schema validation. `score` with any missing candidate score returns
  `UNRESOLVED/rank_score_missing` instead of treating unknown as zero.
- ORDINAL tie detection compares only primary y for vertical order and primary x for
  horizontal order. EXTREME uses the same primary-axis rule; equal extrema are AMBIGUOUS.
- SUBREGION remains a rectangular `scope + reference` construction. `LEFT_SIDE`,
  `RIGHT_SIDE`, `ABOVE`, `BELOW`, `INSIDE`, and `AROUND` are supported. `OUTSIDE` and
  `BOTH_SIDES` remain UNRESOLVED in v1; polygon complements are out of scope.
