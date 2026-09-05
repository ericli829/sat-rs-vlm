# Hard Planner Improvement Backlog

## Scope and ordering principle

This backlog targets semantic planning failures after canonical DSL syntax is
handled by constrained decoding. It is a design plan, not an authorization to
change training data or retrain a model. Items are ordered by expected semantic
gain, engineering cost, and inference-time cost.

| Priority | Work item | Expected gain | Engineering cost | Runtime cost |
|---|---|---:|---:|---:|
| P0 | Hard-error taxonomy and deterministic graph-equivalence metrics | High diagnostic value; prevents optimizing a misleading exact metric | Medium | None during inference |
| P1 | Hard-heavy curriculum plus relation-attachment minimal pairs | High for relational count/object relation | Medium-high | None |
| P1 | Route endpoint decomposition data | Medium-high for route planning | Medium | None |
| P2 | Selective 2/3-shot validated-example retrieval | Medium on nested relations | Medium | Prompt tokens and retrieval latency on routed hard cases |
| P2 | Grammar-valid N-best with validator filtering/reranking | Medium-high if ambiguity dominates | High | 2-4x decoding only on routed hard cases |
| P3 | LoRA target/rank ablation | Unknown until P0/P1 evidence exists | Medium | None after training |
| P4 | Partial or full fine-tuning | Unknown; highest operational risk | High | Same inference, much higher training cost |

## P0: hard-error taxonomy

Every wrong prediction should receive one primary cause and optional secondary
causes. The taxonomy is:

- wrong, missing, or extra operator;
- wrong canonical relation or reversed subject/reference;
- wrong reference attachment or node dependency;
- wrong region or TargetSpec category/attributes;
- wrong SELECT mode, rank, ordinal, extreme, or subregion argument;
- wrong COUNT source/scope/`entire` value;
- wrong final source or answer type;
- wrong or missing residual question;
- harmless non-canonical variation;
- semantically equivalent graph with a different serialization/decomposition.

For `RELATIONAL_COUNT`, additionally record the first broken link in the
referring-expression chain, relation direction, target/reference reversal,
`SELECT_REL` versus `RELATION` confusion, and the node actually counted. For
`ROUTE_PLANNING`, separate endpoint decomposition errors from route-topology
reasoning; the small Planner should produce start, goal, `BUILD_ROUTE_CONTEXT`,
and a residual final, not solve the route.

## P0: layered equivalence evaluation

Keep canonical exact as the strict metric and add two independent layers:

1. **Graph structural exact** after canonical node renaming and safe ordering of
   independent branches. Operator, dependency, input role, enum, final source,
   and answer type must still agree.
2. **Semantic-equivalent pass** after conservative TargetSpec lexical/alias
   normalization and approved residual-question normalization.

Do not collapse opposite relations, change relation attachment, ignore COUNT
scope, or treat extra/dead nodes as equivalent. Emit a reason code and both the
raw and normalized graph for every newly accepted pair. Build a reviewed alias
table rather than using an LLM judge in the first version.

## P1: hard-heavy curriculum and minimal pairs

Construct a topology-balanced training slice rather than adding more natural-
frequency easy examples. Prioritize `RELATIONAL_COUNT`, `OBJECT_RELATION`,
`ROUTE_PLANNING`, and `COMPLEX_REASONING`, stratified by 4-6, 7-10, and 10+
nodes and by relation-chain depth.

Generate reviewed minimal pairs whose wording changes as little as possible
while exactly one graph decision changes:

- left/right, top/bottom, inside/next-to, first/second;
- A relative to B versus B relative to A;
- count A next to B versus count B next to A;
- reference attachment to the immediate object versus an outer container;
- the same candidates with `SELECT_REL`, ordinal, rank, and extreme modes.

For nested relation attachment, use progressive examples: base object, one
relation, two relations, then full chains such as an object inside a U-shaped
road near water next to a playground. Require the graph to expand one verified
dependency at each level. Hold out minimal-pair families, not random rows, to
avoid lexical leakage.

## P1: route endpoint decomposition

Review route failures against semantic equivalence first. Add training cases
only where start/goal referring expressions or their attachment are wrong.
Targets should remain a small stable skeleton:

```text
LOCATE/REGION start
LOCATE/REGION goal
BUILD_ROUTE_CONTEXT(image,start,goal)
FINAL_QUESTION(...)
```

Do not add road-graph search or path selection operators to the Planner target.

## P2: selective validated-example retrieval

Start with BM25 over normalized questions and metadata. Route retrieval only to
the four hard intents or to low-confidence cases. Compare no retrieval, two
examples, and three examples. Retrieved examples must have passed the current
schema/graph/type/semantic validators and should be selected for structural
similarity, relation direction, and graph depth—not answer overlap. Report hard-
task accuracy, prompt tokens, P50/P95 latency, and contamination checks before
considering embedding retrieval.

## P2: N-best and validator reranking

For routed hard cases, generate two to four grammar-valid candidates. Reject
schema, graph, type, and semantic failures first. If several candidates remain,
evaluate simple structural priors before introducing a learned or Teacher
verifier. Never spend multiple generations on stable simple count or
classification cases. Measure oracle@N, selected accuracy, latency, and failure
modes; implement reranking only if oracle@N materially exceeds top-1.

## P3/P4: capacity experiments

Run LoRA rank and target-module ablations only after the P0 metrics and P1 data
show a residual capacity bottleneck. Compare matched data, steps, seeds, and
decoding. Consider partial/full fine-tuning only when hard-validation learning
curves and ablations demonstrate that LoRA—not ambiguity, label noise, or
coverage—is limiting performance.

## Promotion gates

A backlog item is promoted only if it has a frozen hard-test split, exact
provenance, per-intent and per-taxonomy reporting, no regression on stable easy
tasks, and a measured runtime/training-cost budget. Canonical exact, structural
exact, and semantic-equivalent pass must always be reported together.
