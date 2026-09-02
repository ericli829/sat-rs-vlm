# TaskGraph v1.1 Planner DSL

## 1. Purpose

The TaskGraph Planner DSL is a deterministic, model-facing serialization of an
accepted canonical TaskGraph v1.1 `PlannerTarget`. It is not a second semantic
IR, executable Python, or arbitrary code. It can express only operators and
fields already legal in the canonical schema.

The representation boundary is:

```text
Strong Teacher -> canonical JSON -> existing validation -> canonicalization
               -> deterministic DSL compiler -> small-Planner target

Small Planner -> DSL -> restricted parser -> canonical JSON
              -> existing validation -> Executor
```

Canonical JSON remains the only source of truth. The Executor, Validator, and
Capability Router do not consume this DSL directly.

The serialization version is `taskgraph-v1.1-dsl-v1`. It does not change the
TaskGraph semantic schema or the Teacher prompt version
`taskgraph-v1.1-residual-final`.

## 2. Round-trip invariants

For every accepted graph `g`:

```python
canonicalize_target(parse_taskgraph_dsl(compile_taskgraph_to_dsl(g))) \
    == canonicalize_target(g)
```

For canonical DSL `d`:

```python
compile_taskgraph_to_dsl(parse_taskgraph_dsl(d)) == d
```

The parser tolerates insignificant whitespace and newlines. The compiler emits
one node per line, no insignificant whitespace, deterministic TargetSpec
attribute ordering, and exactly one final statement on the last line.

## 3. Grammar

The implementation uses a restricted tokenizer and recursive-descent parser;
it never calls `eval`, `exec`, or a Python AST evaluator.

```ebnf
program       = [intent], node, {node}, final;
intent        = "INTENT", "(", enum, ")";
node          = node_id, "=", call;
final         = ("FINAL" | "FINAL_QUESTION"), "(", arguments, ")";
call          = identifier, "(", [arguments], ")";
arguments     = argument, {",", argument};
argument      = [identifier, "="], value;
value         = reference | string | number | boolean | "null" | enum | list | call;
list          = "[", [value, {",", value}], "]";
reference     = "$image", digit, {digit} | "$n", nonzero_digit, {digit};
node_id       = "n", nonzero_digit, {digit};
boolean       = "true" | "false";
string        = JSON-string;
```

The generic grammar describes syntax only. Lowering accepts nested calls only
where the operator contract permits them (`T(...)` inside `LOCATE`/`COUNT`). It
also rejects unknown named arguments and wrong literal types.

An optional canonical intent is serialized first:

```text
INTENT(RELATIONAL_COUNT)
```

Intent is retained because the current codebase has no exact deterministic
intent inference. If canonical JSON omits intent, DSL omits the `INTENT` line.

## 4. Literals and escaping

- Strings use JSON-compatible double quotes and JSON escaping. The compiler
  uses `json.dumps(..., ensure_ascii=False)`.
- Booleans and null are `true`, `false`, and `null`.
- Integers and finite decimal/exponent numbers use JSON number syntax.
- Lists use comma-separated values without canonical whitespace, for example
  `[1,2,3]`, `[$n1,$n2]`, and `[10000,10000]`.
- Non-finite floats, malformed escapes, trailing commas, and arbitrary mapping
  literals are rejected.
- Canonical system placeholders such as `$question` and `$choices` are schema
  parameter strings, so they appear as quoted strings (`"$question"` and
  `"$choices"`). They are not DSL references.

Unicode text is emitted directly and round-trips unchanged. Quotes, reverse
slashes, newlines, and tabs follow JSON string escaping.

## 5. References and node order

Only `$image0`, `$image1`, ... and `$n1`, `$n2`, ... are reference tokens.
Arbitrary variables such as `$building` or `$result` are invalid.

Node assignments are contiguous and ordered:

```text
n1=...
n2=...
```

The parser rejects duplicate IDs, missing sequence numbers, forward references,
unknown references, nodes after the final statement, and multiple finals. It
then calls the existing TaskGraph validator rather than reimplementing graph,
type, or semantic validation.

## 6. TargetSpec

TargetSpec uses an explicit constructor:

```text
T("ship")
T("ship",size="large")
T("car",color="deep red")
T("building",color="white",has_part="sloped roof")
```

The first positional value is the category. Attributes are named scalar
literals. The compiler writes attribute keys in lexical order. Supported keys
and scalar constraints remain defined exclusively by `schema.py`; the DSL does
not add TargetSpec semantics.

## 7. Canonical operator mapping

| Canonical operator | Canonical DSL surface | Positional/named contract |
|---|---|---|
| `REGION` | `REGION` | `(image,position)` |
| `REGION_FROM_BBOX` | `REGION_BBOX` | `(image,bbox[,image_size])` |
| `FIND_MARKER` | `FIND_MARKER` | `(image,shape[,color])` |
| `LOCATE` | `LOCATE` | `(image,T(...))` |
| `SELECT` / `RELATION` mode | `SELECT_REL` | `(candidates,reference-or-null,relation)` |
| `SELECT` / `RANK` mode | `SELECT_RANK` | `(candidates,reference-or-null,criterion,rank,order)` |
| `SELECT` / `ORDINAL` mode | `SELECT_ORD` | `(candidates,reference-or-null,index,order)` |
| `SELECT` / `EXTREME` mode | `SELECT_EXTREME` | `(candidates,reference-or-null,direction)` |
| `SELECT` / `SUBREGION` mode | `SELECT_SUBREGION` | `(candidates,reference-or-null,subregion)` |
| `GROUP` | `GROUP` | `(entities,mode)` |
| `COUNT` | `COUNT` | `(source,T(...),entire)` when source role is statically unique |
| `COUNT` | `COUNT_IMAGE` | explicit `(image,T(...),entire)` fallback |
| `COUNT` | `COUNT_ENTITIES` | explicit `(entities,T(...),entire)` fallback |
| `ATTRIBUTE` | `ATTRIBUTE` | `(entity,attribute[,part])` |
| `CLASSIFY` | `CLASSIFY` | `(input[,label_space])` |
| `MULTILABEL_CLASSIFY` | `MULTILABEL_CLASSIFY` | `(input,label_space)` |
| `MOTION` | `MOTION` | `(input)` |
| `RELATION` | `RELATION` | `(subject,reference)` |
| `ABS_DIFF` | `ABS_DIFF` | `(a,b)` |
| `VLM_REASON` | `VLM_REASON` | named `image`, `evidence`, required `question`, optional `choices` |
| `BUILD_ROUTE_CONTEXT` | `BUILD_ROUTE_CONTEXT` | `(image,start,goal)` |
| `ROUTE_REASON` | `ROUTE_REASON` | `(context,question,choices)` |
| `MATCH_CHOICE` | `MATCH_CHOICE` | `(value,choices)` |

All enum values remain unquoted canonical enum symbols. Free strings and label
spaces remain quoted/list literals.

### SELECT lowering

The five SELECT surface names are serialization aliases, not new semantic
operators. For example:

```text
n4=SELECT_REL($n3,$n2,NEXT_TO)
```

lowers to:

```json
{
  "id": "n4",
  "op": "SELECT",
  "inputs": {"candidates": "$n3", "reference": "$n2"},
  "params": {"mode": "RELATION", "relation": "NEXT_TO"}
}
```

The reference slot is always present in SELECT DSL. `null` represents an absent
canonical `inputs.reference`, which prevents argument-count ambiguity.

### COUNT source role

`COUNT($source,T(...),true|false)` is used only when current static output types
identify exactly one canonical input role (`image` or `entities`). Canonical
`SELECT` has the union output type `Entity | EntitySet | Region | RegionSet`, so
its role is not statically unique. In that case the compiler emits
`COUNT_IMAGE` or `COUNT_ENTITIES` according to the canonical input key. The
parser never guesses.

## 8. Final statements

Structured final:

```text
FINAL($n5,CHOICE_SINGLE)
FINAL([$n1,$n2],CHOICE_MULTI)
```

Residual question final:

```text
FINAL_QUESTION([$n1,$n2],CHOICE_SINGLE,"What is the likely purpose of these ponds?")
```

These lower directly to canonical `final.sources`, optional `final.question`,
and `final.answer_type`. The compiler/parser does not decide whether a question
is semantically appropriate. The existing residual-final validator retains
that responsibility.

## 9. Examples

### Simple COUNT: JSON to DSL

```json
{
  "intent": "SIMPLE_COUNT",
  "nodes": [{
    "id": "n1",
    "op": "COUNT",
    "inputs": {"image": "$image0"},
    "params": {
      "target": {"category": "ship", "attributes": {"size": "large"}},
      "entire": true
    }
  }],
  "final": {"sources": ["$n1"], "answer_type": "CHOICE_SINGLE"}
}
```

```text
INTENT(SIMPLE_COUNT)
n1=COUNT($image0,T("ship",size="large"),true)
FINAL($n1,CHOICE_SINGLE)
```

### Relational count

```text
INTENT(RELATIONAL_COUNT)
n1=REGION($image0,TOP)
n2=LOCATE($n1,T("building"))
n3=LOCATE($image0,T("sun umbrella",color="red"))
n4=SELECT_REL($n3,$n2,NEXT_TO)
n5=COUNT_ENTITIES($n4,T("sun umbrella",color="red"),false)
FINAL($n5,CHOICE_SINGLE)
```

`COUNT_ENTITIES` is necessary here because the canonical static type of a
SELECT result is intentionally a union containing both visual and entity-set
possibilities.

### Route residual final

```text
INTENT(ROUTE_PLANNING)
n1=LOCATE($image0,T("roundabout"))
n2=LOCATE($image0,T("pond"))
n3=BUILD_ROUTE_CONTEXT($image0,$n1,$n2)
FINAL_QUESTION($n3,CHOICE_SINGLE,"Which option describes the best route between the selected start and goal?")
```

### Multi-source terminal semantic reasoning

```text
INTENT(COMPLEX_REASONING)
n1=LOCATE($image0,T("pond"))
n2=LOCATE($image0,T("farmland"))
FINAL_QUESTION([$n1,$n2],CHOICE_SINGLE,"What is the most likely purpose of these ponds in this agricultural setting?")
```

## 10. Error behavior and trust boundary

Tokenizer or grammar failures raise `DSLParseError` with a source location.
Unknown operators/enums, malformed TargetSpec values, unsafe identifiers,
duplicate/forward references, missing finals, and type/semantic failures fail
closed. There is no fuzzy repair and injection-like text such as
`__import__("os")`, `eval(...)`, or `system(...)` is rejected as an unknown
operator and never executed.

Compiler input is canonicalized and passed through the existing validator.
Invalid graphs raise `DSLCompileError`; fields are never silently dropped to
make a graph serializable.

## 11. Developer API and CLI

```python
from taskgraph_lab.taskgraph.dsl import (
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
```

```powershell
python -m taskgraph_lab.taskgraph.dsl compile graph.json
python -m taskgraph_lab.taskgraph.dsl parse graph.dsl
```

The compile command prints canonical DSL. The parse command prints pretty
canonical JSON after existing validation.

## 12. Constrained decoding

`CanonicalDSLPrefixGrammar` recognizes every prefix of the canonical compiler
language. `GreedyDSLLogitsProcessor` uses it during deterministic generation
and selects the highest-logit token that keeps the continuation in that
language. The compiler-language regression invariant is:

```python
grammar.accepts(compile_taskgraph_to_dsl(graph))
```

The decoder hard-constrains operator surfaces and arity, canonical enums,
punctuation, JSON literals and string escaping, contiguous `n1`, `n2`, ...
assignments, caller-provided image references, references to preceding nodes,
TargetSpec shape, and the two final forms. A complete `FINAL` or
`FINAL_QUESTION` forces EOS on the next generation step. Optional node-budget
and repeated-node-cycle guards stop pathological continuations and report the
termination reason; they are operational limits rather than additions to the
canonical DSL language.

The constraint deliberately does not decide operator choice, relation meaning,
reference attachment, TargetSpec wording, graph topology, count scope, residual
question semantics, output types, or whether every node contributes to the
final dependency chain. Parsing plus existing schema, graph, type, and semantic
validation remain mandatory after generation.

The current Qwen/Transformers path uses the custom processor instead of
XGrammar. XGrammar is not part of the stable environment, the installed
Transformers version exposes no native grammar argument, and request-scoped
node/image references still require dynamic state. Constraint failures fail
closed and are recorded; there is no silent unconstrained fallback.
