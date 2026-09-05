# Answerability

Answerability is an optional evidence-sufficiency service attached to `TaskGraphRuntime`.
It is deliberately outside the logical DAG: it cannot add graph nodes, stop traversal, or
decide whether exhaustive `COUNT` should continue.

`EvidenceSufficiencyExecutor` composes the supplied typed evidence with `InputComposer` and
calls the existing `semantic_2b.reason_and_decide` finite-decision primitive. The underlying
Qwen provider performs one visual/text reasoning prefill, scores the status continuation from
that same KV session, and releases the session in the provider's `finally` path. There is no
second model, cache implementation, or persistent cross-request tensor cache.

The public result status is one of:

- `SUFFICIENT`
- `NEED_MORE_EVIDENCE`
- `UNRESOLVED`
- `ERROR`

Results contain confidence, reason code, provider/model/method, cache reuse, latency, and a
SHA-256 fingerprint over sample/question/evidence version and trace-safe source summaries.
They never contain free reasoning text, image bytes, tensors, or cache/session objects. Cache
ownership is request-scoped and provider metadata must report session release on a real run.

`scripts/taskgraph/benchmark_answerability_cache.py` compares cache-on and cache-off
cross-stage execution. It reports prefill tokens, reused tokens, total/wall latency, peak VRAM,
visual prefill count, and an explicit null TTFT because the current backend does not expose
first-token timing. Model paths must be local; the script never downloads checkpoints.
