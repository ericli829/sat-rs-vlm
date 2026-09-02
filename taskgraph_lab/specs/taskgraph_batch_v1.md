# TaskGraph Teacher batch transport v1

`taskgraph-batch-v1` is an API transport envelope only. It does not change the
TaskGraph v1.1 semantic schema or Planner DSL.

## Request

```json
{
  "mode": "batch",
  "samples": [
    {
      "sample_id": "sample-a",
      "question": "...",
      "question_type": "MULTIPLE_CHOICE",
      "choices": ["..."],
      "inputs": {"image0": {"type": "image", "uri_or_key": "..."}},
      "metadata": {}
    }
  ]
}
```

## Response

```json
{
  "batch_version": "taskgraph-batch-v1",
  "results": [
    {
      "sample_id": "sample-a",
      "taskgraph": {
        "intent": "...",
        "nodes": [],
        "final": {}
      }
    }
  ]
}
```

Each `taskgraph` is parsed, canonicalized, validated, compiled to DSL, parsed
back, and compared independently. Missing, duplicate, unknown, malformed, and
out-of-order results are recorded explicitly. Valid unique results remain
available even when peers fail.

Failed items alone form a partial-repair batch. Valid peers are never sent
again. A response that cannot be parsed as a batch envelope receives one
transport-only repair; if that fails, the request is recursively bisected down
to singleton requests with bounded retries.
