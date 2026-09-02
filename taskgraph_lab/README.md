# TaskGraph Planner data-generation lab

`taskgraph_lab/` is an isolated experiment for generating and validating
TaskGraph v1.1 supervision for a future text Planner. It is not a production
runtime, does not execute graphs, and does not modify the existing UHR Locator.

The lab remains separate because the schema is not frozen. Teacher labels must
first survive strict validation, one bounded repair attempt, optional semantic
review, and human spot checks. Stable components can be considered for migration
into `src/` later.

## Data flow

```text
raw QA
  -> text-only normalization
  -> structure-coverage seed set
  -> Teacher generation
  -> schema / graph / type / semantic validation
  -> local lexical normalization or one bounded schema repair
  -> optional semantic review
  -> canonical TaskGraph
  -> deterministic Planner DSL serialization
  -> Planner SFT JSONL (canonical JSON plus optional `planner_dsl`)
```

## Final choice contract

Canonical v1.1 targets end with a non-empty `final.sources` list and an
`answer_type`; `final.question` is optional. Authoritative structured finals
omit it when their value maps deterministically to an answer. Visual or
semantic finals include a static residual question that describes only the
judgment left after upstream grounding and computation. Options are never
copied into `final`, and the Planner never writes predicted runtime values into
the residual question.

The conceptual boundary is `ChoiceRequest(sources, question: Optional[str],
options)`, backed by a future `InputComposer`. An absent question selects the
deterministic structured resolver; a present question selects the semantic/VLM
resolver after visual runtime objects are materialized as crops or annotated
views. Structured values retain explicit types, and mixed inputs must not be
flattened with `str()`. These interfaces are specified in
`specs/taskgraph_v1_1_final_choice_contract.md`; no Choice VLM or image composer
is implemented in this lab.

`COUNT` is authoritative structured evidence and normally has no final
question. Its resolver must not automatically re-add the source image and
recount visually.

The four maintained final-choice examples live in
`prompts/few_shot_final_choice.txt` (count, bbox compound color, relational
visual attribute, and route context).

No image bytes are opened, encoded, or sent to the Teacher. Requests contain
only question text, question type, choices, image keys/paths, and dataset/category
metadata. Ground-truth answers are deliberately omitted from normalized samples
and model prompts.

## Layout

- `specs/`: exact copies of the supplied TaskGraph v1/v1.1 design documents.
- `prompts/`: compact Teacher, repair, and semantic-review prompts.
- `taskgraph/`: Pydantic schema, layered validator, type checker, canonicalizer.
- `taskgraph/dsl/`: deterministic JSON/DSL compiler, restricted parser, and CLI.
- `datasets/`: XLRS-Bench and MME RealWorld RS text-only adapters.
- `generation/`: provider abstraction and resumable JSONL generation pipeline.
- `tools/`: seed construction, validation, reporting, and SFT export.
- `tests/fixtures/`: CPU-only, network-free smoke inputs.
- `data/` and `outputs/`: generated local artifacts; ignored by this lab's
  `.gitignore` and created on demand.

## Install and test

Run from the repository root. On the current Windows development machine:

```powershell
$py = 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe'
& $py -m pip install -r taskgraph_lab/requirements.txt
& $py -m pytest taskgraph_lab/tests -q
```

## Normalize real metadata

Both paths are optional individually; at least one is required. They are never
hard-coded as the only dataset location.

```powershell
& $py -m taskgraph_lab.datasets.normalize `
  --xlrs-json 'D:\data\XLRS-Bench\questions_answers.json' `
  --mme-json 'D:\data\MME-RealWorld-RS\MME_RealWorld.json' `
  --output taskgraph_lab/data/normalized/all.jsonl
```

Supported inputs are `.json` arrays, `.json` objects containing a
`records`/`samples`/`data` array, and JSONL. Unexpected formats fail clearly.

## Build a deterministic 200--300 sample seed

The default configuration targets 250 samples and uses structure-aware quotas
instead of plain random sampling:

```powershell
& $py -m taskgraph_lab.tools.build_seed_set `
  --xlrs-json 'D:\data\XLRS-Bench\questions_answers.json' `
  --mme-json 'D:\data\MME-RealWorld-RS\MME_RealWorld.json' `
  --config taskgraph_lab/configs/seed_sampling.yaml `
  --output-dir taskgraph_lab/data/seeds/generated_v1
```

The output contains `seed.jsonl`, `seed_manifest.json`, source SHA256 values,
sample IDs, and `category_distribution.json`. Re-running with identical inputs,
config, and seed produces the same selection.

## Teacher providers

`DryRunProvider` uses no network and records its request payload. The generic
OpenAI-compatible provider uses an explicit chat-completions endpoint and the
standard JSON-object response mode. The TaskGraph schema has no API-SDK
dependency.

Copy `configs/generation.example.yaml` to an ignored local config and choose:

```yaml
provider:
  type: openai_compatible
  endpoint_env: TASKGRAPH_TEACHER_ENDPOINT
  model_env: TASKGRAPH_TEACHER_MODEL
  api_key_env: TASKGRAPH_TEACHER_API_KEY
```

Set secrets only in the environment; never put them in YAML or logs:

```powershell
$env:TASKGRAPH_TEACHER_ENDPOINT = 'https://provider.example/v1/chat/completions'
$env:TASKGRAPH_TEACHER_MODEL = 'teacher-model-name'
$env:TASKGRAPH_TEACHER_API_KEY = 'secret'
```

The endpoint must be the complete OpenAI-compatible chat-completions URL. A
vendor-specific endpoint or field set should be implemented as another provider
rather than guessed in this generic adapter.

For the checked-in `configs/deepseek_v4_flash.yaml`, enter the key interactively
for each command so it is absent from files and shell history.

The maintained DeepSeek generation and batch-benchmark configs explicitly use
thinking mode with `reasoning_effort: low`. Current experiments do not run a
thinking-disabled control. Thinking settings are recorded in provider metadata.

Use:

```powershell
$env:TASKGRAPH_TEACHER_API_KEY = Read-Host 'DeepSeek API key' -MaskInput
try {
  & $py -m taskgraph_lab.generation.generate `
    --input taskgraph_lab/data/seeds/generated_v1/seed.jsonl `
    --config taskgraph_lab/configs/deepseek_v4_flash.yaml `
    --output taskgraph_lab/outputs/raw/deepseek_v4_flash.jsonl
} finally {
  Remove-Item Env:TASKGRAPH_TEACHER_API_KEY -ErrorAction SilentlyContinue
}
```

The original generation command remains a backward-compatible single-item API.
The batch API uses the separate `taskgraph-batch-v1` transport envelope and
keeps every contained TaskGraph canonical v1.1. Batching never combines graph
semantics: IDs, node references, validation, repair, DSL compilation, and DSL
round-trip checks are all scoped per sample.

## Qwen3-VL-4B Planner cloud run

`configs/qwen3vl_4b_planner_lora_cloud.yaml` is the 4B replacement for the
previous 2B Planner. It keeps the vision encoder and projector frozen, trains
only language-model LoRA, uses BF16/SDPA, and raises the single-device training
batch from 1 to 2 with four-step accumulation. The profile targets the
`planner_sft_hard_curriculum_v1` split and its 3,934 training / 217 test rows.

On the cloud host, from the repository root:

```bash
export QWEN3VL_4B_MODEL_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
export OUTPUT_ROOT=/root/autodl-tmp/outputs/taskgraph
nohup bash taskgraph_lab/tools/run_qwen3vl_4b_planner_cloud.sh \
  > /root/autodl-tmp/outputs/taskgraph/qwen3vl_4b_planner.log 2>&1 &
echo $! > /root/autodl-tmp/outputs/taskgraph/qwen3vl_4b_planner.pid
```

The runner starts evaluation only after training exits successfully. Evaluation
uses greedy constrained DSL decoding with bounded recovery and writes its
predictions and summary below the same timestamped run directory.

## Batch Teacher benchmark

`generation.batch_generation.generate_teacher_batch` chunks samples by both a
sample limit and a tokenizer-free input-token budget. It preserves valid items
immediately and sends only failed items through one partial-repair batch. A
catastrophic JSON-wrapper failure gets one transport-only repair attempt and is
then bisected; it never causes already valid items to be regenerated.

The fixed 24-sample benchmark contains 23 source records plus one explicitly
marked synthetic multi-image compiler case. It covers simple and relational
counting, bbox, marker, relation, ordinal/rank, route, complex reasoning, an
8-plus-node stress case, an explicit category alternative, and a source
question-type conflict. Build it deterministically with:

```powershell
& $py -m taskgraph_lab.tools.build_batch_benchmark_seed `
  --source taskgraph_lab/data/seeds/generated_v1/seed.jsonl `
  --output taskgraph_lab/data/seeds/generated_v1/batch_benchmark_24_v1.jsonl `
  --manifest taskgraph_lab/data/seeds/generated_v1/batch_benchmark_24_v1_manifest.json
```

Run batch sizes 1, 2, 4, and 8 in thinking-low mode with the interactive
background launcher:

```powershell
& taskgraph_lab/tools/start_batch_benchmark.ps1 -PythonPath $py
```

Each run retains per-item raw/valid/repaired/rejected records, call traces, token
usage, transport diagnostics, and batch provenance. The final JSON and Markdown
reports compare calls/sample, accepted/call, prompt and completion tokens/sample,
repair calls, transport failures, validation rates, DSL success, and latency.

## Generation, resume, repair, and review

Dry-run the full pipeline first:

```powershell
& $py -m taskgraph_lab.generation.generate `
  --input taskgraph_lab/tests/fixtures/normalized_samples.jsonl `
  --config taskgraph_lab/configs/generation.example.yaml `
  --output taskgraph_lab/outputs/raw/smoke.jsonl
```

For a real seed, replace `--input` and use a local config whose provider type is
`openai_compatible`. Each record is appended and flushed immediately. Existing
`sample_id` values in the raw output are skipped automatically, so the same
command resumes safely after interruption. Provider/model/timestamp/latency,
token usage, attempt, prompt version, and schema version are retained.

Safe lexical aliases and legacy `GraphNode.output` are normalized locally and
recorded in `normalized_fields`. Malformed JSON and straightforward schema
defects trigger at most one repair when `runtime.repair_enabled: true`.
Dependency/type/semantic planning errors are rejected without LLM repair so the
repair pass cannot silently relabel a bad plan. Warnings never trigger repair or
rejection. Failed repair enters `outputs/rejected/`; successful repair enters
`outputs/repaired/`.

Semantic review is off by default. Enable it with either
`runtime.semantic_review: true` or `--semantic-review`. Review records use
`VALID`, `VALID_BUT_NON_MINIMAL`, `SEMANTICALLY_AMBIGUOUS`, or `LOGIC_ERROR` and
never rewrite a graph.

Concurrency, requests/minute, timeout, maximum retries, exponential backoff,
temperature, and output-token limits are configured under `runtime`.
Accepted records include compiler-derived `planner_dsl` by default. Set
`runtime.emit_planner_dsl: false` to omit it; the Teacher response remains JSON
in either mode.

## Validate and summarize

```powershell
& $py -m taskgraph_lab.tools.validate_jsonl `
  --input taskgraph_lab/outputs/valid/smoke.jsonl `
  --output taskgraph_lab/outputs/reports/smoke_validation.jsonl

& $py -m taskgraph_lab.tools.summarize `
  --raw taskgraph_lab/outputs/raw/smoke.jsonl `
  --valid taskgraph_lab/outputs/valid/smoke.jsonl `
  --repaired taskgraph_lab/outputs/repaired/smoke.jsonl `
  --rejected taskgraph_lab/outputs/rejected/smoke.jsonl `
  --reviews taskgraph_lab/outputs/reviews/smoke.jsonl `
  --output-dir taskgraph_lab/outputs/reports/smoke
```

Reports include generation/API/internal-processing failures, layered validation
rates, repair rate, warnings/errors, operator and intent frequencies, node
counts, dataset/category coverage, and review verdicts.

## Export Planner SFT data

Framework-neutral target format:

```powershell
& $py -m taskgraph_lab.tools.export_sft `
  --input taskgraph_lab/outputs/valid/smoke.jsonl `
  --input taskgraph_lab/outputs/repaired/smoke.jsonl `
  --output taskgraph_lab/outputs/sft/smoke.jsonl
```

Messages format is available with `--format messages`. Supply `--reviews` to
exclude ambiguous or logically incorrect reviewed samples; add
`--allow-non-minimal` only after accepting that policy explicitly.

## Compile and inspect Planner DSL

The DSL is generated only after canonical validation. It is never requested
from the Teacher:

```powershell
& $py -m taskgraph_lab.taskgraph.dsl compile graph.json
& $py -m taskgraph_lab.taskgraph.dsl parse graph.dsl
```

The complete grammar, operator mapping, reversible COUNT fallback, and
round-trip invariant are documented in `specs/taskgraph_v1_1_dsl.md`.

## Explicit non-goals

This lab does not implement a TaskGraph Executor, detector, CLIP/VisRAG,
multiscale COUNT, route VLM, answerability model, model routing, or Planner
training.
