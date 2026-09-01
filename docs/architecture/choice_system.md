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
does not receive `pixel_values` again. `CHOICE_SINGLE` consumes the reasoning cache in
place while appending its suffix, so the common single-token path does not deepcopy the
full reasoning KV. If a legal label itself spans multiple tokens, only the necessary
candidate continuations fork the suffix-extended cache. `CHOICE_MULTI` forks the original
reasoning cache once per independent option verification; these forks are sequential and
correct but not yet batched.

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
- `CHOICE_SINGLE`: append `choice.single_choice_suffix` (`Final choice:` by default), then
  select exactly one legal ID by score argmax.
- `CHOICE_MULTI`: render `choice.multi_verify_template` independently for each original
  option and use
  `score(YES) - score(NO)`. Select every score above
  `choice.multi_select_threshold`, preserving original option order. Zero selections remain
  empty or unresolved according to `choice.multi_empty_policy`; the runtime never inserts
  an argmax answer.

`YES` and `NO`, like option IDs, use a single-token fast path only when the actual tokenizer
produces one token; otherwise their complete continuations are scored.

The unified `ChoiceScoreResult` preserves `selected_ids`, per-ID scores, answer type,
reasoning text, provider/model, method, cache reuse, latency, and scalar metadata. The
final `ChoiceResult` also stores `answer_type`. Its legacy `choice_id` property is populated
only for `CHOICE_SINGLE` with exactly one selected ID. `CHOICE_MULTI` may contain exactly
one selected option and still remains `CHOICE_MULTI`, with `choice_id: null`; multi-choice
is never flattened to a string such as `"A,C"`.

Trace summaries preserve the distinction explicitly:

```json
{"answer_type": "CHOICE_SINGLE", "selected_ids": ["C"], "choice_id": "C"}
```

```json
{"answer_type": "CHOICE_MULTI", "selected_ids": ["C"], "choice_id": null}
```

```yaml
choice:
  backend: kv_cached_logits
  single_choice_suffix: "\n\nFinal choice:"
  multi_verify_template: |-
    Verify the following option independently.

    Candidate option {choice_id}: {option_text}
    Is this option a correct answer to the original question?
    Answer YES or NO:
  legacy_regex_fallback: false
  multi_select_threshold: 0.0
  multi_empty_policy: EMPTY
  preserve_reasoning_text: true
```

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
- `reasoning_total_ms` records that same currently measurable reasoning wall time;
- `cache_clone_ms`, `suffix_tokenize_ms`, `choice_suffix_prefill_ms`, and
  `choice_scoring_ms` retain useful detail;
- `choice_total_ms` starts on entry to cached choice scoring and includes validation,
  tokenization, Python setup, cache clone/fork, suffix forward, and candidate scoring;
- `total_ms` is `reasoning_total_ms + choice_total_ms`, and benchmarks report
  `choice_total_ms / reasoning_total_ms`;
- token counts include initial prefill, reasoning, suffix, and scored continuation tokens;
- model ID, device, dtype, cache reuse, and peak allocated VRAM (when CUDA exposes it) are
  scalar metadata.

Run the lightweight current-architecture benchmark without downloads:

```powershell
python scripts/taskgraph/benchmark_choice_cache.py `
  --model-id $env:QWEN3VL_2B_MODEL_DIR `
  --image path/to/image.png `
  --prompt "Which scene type is shown?" `
  --options-json '["A urban","B water","C farmland"]' `
  --output outputs/taskgraph/choice-cache.json
```

For Route, pass the 4B model as `--model-id` and set `--role route_4b`. This measures only
the frozen 4B reasoning to same-4B cached choice flow. There is no 4B-versus-2B Route
architecture comparison.

Thresholds require dataset-level calibration. The default `0.0` corresponds to preferring
`YES` over `NO`; it is a contract default, not a claim of universal probability calibration.

## Validated dependency environment

The cache API and unit compatibility checks in this branch were validated with:

```text
torch 2.6.0+cu126
transformers 5.13.0
```

The declared Transformers dependency range remains unchanged; this validation records a
known working environment rather than claiming a complete version matrix.
