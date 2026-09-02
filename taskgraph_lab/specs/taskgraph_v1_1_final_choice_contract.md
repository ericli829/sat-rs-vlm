# TaskGraph v1.1 final-choice and multi-source contract

## Scope

The Planner sees only the original question, original choices, and input
metadata. It produces a static logical DAG; it never sees or predicts runtime
node values. This document defines interfaces only. It does not implement a
Choice VLM, visual materialization, or an Executor.

## Canonical `FinalSpec`

Structured final (the upstream DAG already produced an authoritative value):

```json
{
  "sources": ["$n3"],
  "answer_type": "CHOICE_SINGLE"
}
```

Question final (the final sources still require semantic or visual judgment):

```json
{
  "sources": ["$n3", "$n7"],
  "question": "Which option best explains the observed pattern in these regions?",
  "answer_type": "CHOICE_SINGLE"
}
```

- `sources` is a non-empty, duplicate-free list of existing `$nX` references.
- Raw `$imageX` references are forbidden as final sources.
- `question` is optional. When present, it is non-empty static text generated
  before execution.
- Structured outputs such as `ScalarInt`, `Boolean`, `Label`, and `LabelSet`
  normally omit `question` when they can be mapped to an answer
  deterministically.
- Visual or semantic outputs such as `Region`, `RegionSet`, `Entity`,
  `EntitySet`, `RouteContext`, and `EvidenceSet` require a residual `question`
  that states only the unresolved final judgment.
- A residual `question` must not contain predicted counts, labels, booleans, or
  other runtime results. Referring expressions already resolved by the DAG
  should be removed.
- `answer_type` uses `CHOICE_SINGLE`, `CHOICE_MULTI`, `INTEGER`, `BOOLEAN`,
  `LABEL`, `LABEL_SET`, or `TEXT`; runtime types such as `Answer` and
  `ScalarInt` are not answer types.
- `choices` is not a `FinalSpec` field. The Planner must not copy, reorder, or
  rewrite dataset options.

The compatibility loader accepts legacy `source: "$nX"` and canonicalizes it
to `sources: ["$nX"]`. It does not invent a generic final question: absence is
meaningful for an authoritative structured final.

## Future runtime contract

```text
ChoiceRequest {
    sources: list[RuntimeObject],
    question: Optional[str],
    options: list[str],
    answer_type: CHOICE_SINGLE | CHOICE_MULTI
}
```

- `sources` is resolved from `final.sources` by `RuntimeStore`.
- `question` is exactly `final.question`, or `None` when it is absent.
- `options` is exactly the original dataset choices, unchanged.
- `CHOICE_MULTI` is an indeterminate multi-select contract: select every
  applicable option and return one or more option ids. A particular sample may
  legitimately have only one selected id.
- `CHOICE_SINGLE` requires exactly one selected id.

The final Choice VLM boundary must state this cardinality contract explicitly.
The canonical machine response is `{"choice_ids":["A","C"]}` for
`CHOICE_MULTI` and `{"choice_ids":["A"]}` for `CHOICE_SINGLE`; prose-only or
ambiguous "return an option" instructions are forbidden.

When `question` is absent, a deterministic resolver maps structured sources to
the answer or original options. When it is present, `InputComposer` materializes
the sources and a semantic/VLM resolver handles the residual question. Runtime
must not re-add the original image merely because `question` is absent.

`RuntimeStore` retains typed objects, for example:

```text
$n1 -> Region
$n2 -> EntitySet
$n3 -> ScalarInt
$n4 -> Label
```

Objects are not converted with `str()` while passing between nodes. A future
boundary interface is:

```text
InputComposer.compose(
    sources: list[RuntimeObject],
    question: str | None,
    choices: list[str] | None
) -> ModelInput
```

Visual types (`Region`, `Entity`, `EntitySet`, `RouteContext`) are materialized
as crops, annotated views, or other visual payloads. Structured types
(`ScalarInt`, `ScalarFloat`, `Boolean`, `Label`, `LabelSet`) are serialized with
their type and value, for example:

```text
[result_1]
type: ScalarInt
value: 7
```

For mixed sources, visual and structured sections remain separate until the
composer creates `ModelInput`; direct string concatenation is forbidden.

## Counting authority

A `COUNT` result is authoritative structured evidence:

```json
{
  "sources": ["$n1"],
  "answer_type": "CHOICE_SINGLE"
}
```

The deterministic final resolver receives the resolved `ScalarInt` and original
options. It must not automatically reintroduce the original image or visually
recount it. A final source set mixing an authoritative count branch with visual
evidence is rejected by the current validator.

## Visual choice

Compound visual options may require a selected visual object instead of a
single `ATTRIBUTE` label:

```json
{
  "sources": ["$n4"],
  "question": "What colors are visible on the selected house?",
  "answer_type": "CHOICE_SINGLE"
}
```

Here `$n4` may be an `Entity`, `EntitySet`, or `Region` produced after the DAG
has resolved the long referring expression. `ATTRIBUTE` is not mandatory when
the final visual choice requires a compound judgment.

## Named and variable-length inputs

Fixed semantic roles remain named mappings:

```json
{"subject": "$n2", "reference": "$n4"}
{"a": "$n2", "b": "$n4"}
{"image": "$image0", "start": "$n2", "goal": "$n4"}
{"candidates": "$n3", "reference": "$n1"}
```

Anonymous lists are forbidden for fixed-role operators. Only a role whose
operator semantics explicitly permits variable evidence may use `list[Ref]`.
TaskGraph v1.1 currently permits this only for `VLM_REASON.evidence`:

```json
{"image": "$image0", "evidence": ["$n2", "$n4", "$n7"]}
```

The type checker validates and records every list element independently.

## Static validation

Validation rejects empty/duplicate final sources, missing refs, raw image final
refs, blank questions when the optional key is present, visual/route finals
without a residual question, obvious runtime-result leakage, copied option
lists, MCQ answer-type mismatches, dead nodes relative to the union of all
final sinks, and visual reintroduction alongside an authoritative count.
Generic choice questions attached to authoritative structured outputs are
accepted for compatibility but reported as non-minimal warnings.

## Planner serialization

Accepted canonical targets may be deterministically serialized for small
Planner training. The DSL is only a surface syntax and does not change this
contract; see `taskgraph_v1_1_dsl.md`.
