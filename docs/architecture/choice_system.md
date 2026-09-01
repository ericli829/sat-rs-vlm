# Choice System

## Responsibility and non-goals

The Choice System converts typed evidence or visual reasoning into legal benchmark option
IDs. It supports `CHOICE_SINGLE` and indeterminate `CHOICE_MULTI` without changing the
dataset's original option text. It does not train a Choice LoRA, create a new MCQ dataset,
or load a separate Choice model by default. The existing instruction-tuned Qwen3-VL
models already provide the needed reasoning and token probabilities; the runtime adds a
deterministic constrained decision layer.

Free reasoning text is never parsed on the production path. A response such as
`A looks possible ... therefore C` is only reasoning. Final IDs come from legal-token
continuation scores. The old exact-text parser exists only behind
`choice.legacy_regex_fallback: true`, is disabled by default, and supports single-choice
compatibility only.

## Frozen flows

### General 2B and direct multiple choice

```text
Qwen3-VL-2B visual/text prefill
  -> free reasoning decode
  -> retain the same model's past_key_values
  -> append the short final suffix
  -> score only legal option continuations
  -> deterministic ChoiceScoreResult
```

`providers.choice.reuse: semantic_2b` makes `choice` and `semantic_2b` the same provider
object, therefore the same lazy `HuggingFaceVLMEngine`, model, and processor. An explicitly
configured independent provider remains possible for future experiments.

### Route 4B

`ROUTE_REASON` sends the marked `RouteContext`, original question, and original options to
`route_4b.reason_and_choose`. The 4B session performs both route reasoning and constrained
choice. The node stores a tensor-free `ChoiceScoreResult`; the final resolver consumes it
without calling the 2B provider. A 4B cache is never passed to 2B (or vice versa): every
`CachedGenerationSession` carries a model identity and cross-model use raises an error.

### SELECT

Fuzzy relation SELECT uses the local candidate canvas and stable candidate IDs. With
`selection_type: SINGLE`, legal IDs are scored by argmax. With `selection_type: MULTI`
(the compatibility default), each candidate independently forks the same reasoning prefix
and scores `YES` versus `NO`. The configured threshold selects candidates, and selected IDs
map back through the canvas's ID-to-Entity metadata. Neither path parses reasoning text or
repeats visual prefill.

## Engine cache API and lifecycle

`HuggingFaceVLMEngine.reason_with_cache` calls Transformers 5.x `generate` with
`use_cache=True` and `return_dict_in_generate=True`. It returns free reasoning and an opaque
`CachedGenerationSession` containing the actual `past_key_values`, sequence state,
attention state, Qwen M-RoPE delta, and model identity. A terminal EOS selected by reasoning
is removed before the suffix so the constrained step remains in the same assistant turn.

`score_choice_from_cache` validates model identity and active state, then supplies that
cache to `prepare_inputs_for_generation` and the model forward pass. Qwen3-VL therefore
does not receive `pixel_values` again. General-path candidate continuations fork the same
base cache sequentially; this is correct but not yet batched.

`reason_and_choose` is the preferred high-level API. It always closes the session in a
`finally` block. Closing drops every tensor reference and removes the session from the
engine registry. Closed sessions and foreign-model sessions are rejected. No cache tensor
enters `RuntimeStore`, provenance, trace JSON, or benchmark output.

## Option tokenization and scoring

The scorer tokenizes the continuation appropriate to the suffix boundary: for a suffix
ending in punctuation it scores ` A`; for a suffix ending in whitespace it scores `A`.
It never assumes labels are one token.

- Fast path: if every legal ID is one token, select from the next-token logits.
- General path: fork the same base cache for every legal ID and sum token log-probabilities
  over its complete continuation. Unequal token lengths are length-normalized.
- `CHOICE_SINGLE`: select exactly one legal ID by score argmax.
- `CHOICE_MULTI`: for each original option, append a short verification question and use
  `score(YES) - score(NO)`. Select every score above
  `choice.multi_select_threshold`, preserving original option order. Zero selections remain
  empty or unresolved according to `choice.multi_empty_policy`; the runtime never inserts
  an argmax answer.

The unified `ChoiceScoreResult` preserves `selected_ids`, per-ID scores, answer type,
reasoning text, provider/model, method, cache reuse, latency, and scalar metadata. The
legacy `ChoiceResult.choice_id` property is populated only when exactly one ID is selected;
multi-choice is never flattened to a string such as `"A,C"`.

## Structured deterministic mapping

Before invoking a model, `ChoiceResolver` attempts exact mapping for authoritative
`ScalarInt`, `ScalarFloat`, `Boolean`, `Label`, and `LabelSet` values after removing only the
benchmark option prefix. Examples include count `7 -> "B 7"`, label `red -> "C red"`, and
boolean `true -> yes`. Ambiguous or non-exact mappings fall through to cached 2B scoring.
A precomputed 2B/4B `ChoiceScoreResult` has the next priority and is reused directly.

## Trace and latency

Cached results expose measurable timing without pretending to separate work that the
Transformers generation loop reports jointly:

- `vision_prefill_ms`, `text_prefill_ms`, and `total_prefill_ms` are `null` when unavailable;
- `reasoning_decode_ms` is measured generation time and metadata states that it includes
  prefill;
- `choice_suffix_prefill_ms`, `choice_scoring_ms`, and `total_ms` are measured;
- token counts include initial prefill, reasoning, suffix, and scored continuation tokens;
- model ID, device, dtype, cache reuse, and peak allocated VRAM (when CUDA exposes it) are
  scalar metadata.

Run the lightweight real-model comparison without downloads:

```powershell
python scripts/taskgraph/benchmark_choice_cache.py `
  --model-id $env:QWEN3VL_2B_MODEL_DIR `
  --image path/to/image.png `
  --prompt "Which scene type is shown?" `
  --options-json '["A urban","B water","C farmland"]' `
  --output outputs/taskgraph/choice-cache.json
```

For the historical Route comparison, pass the 4B model as `--model-id`, set
`--role route_4b`, and pass the 2B path as `--legacy-choice-model-id`. The baseline then
measures recomputed 4B reasoning plus a 2B choice request; the cached path measures 4B
reasoning plus the incremental same-4B choice.

Thresholds require dataset-level calibration. The default `0.0` corresponds to preferring
`YES` over `NO`; it is a contract default, not a claim of universal probability calibration.
