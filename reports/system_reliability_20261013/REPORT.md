# Competition inference reliability work — 2026-10-13

## Scope and status

This report covers the local reliability implementation and the teammate's GitHub CLIP integration. GitHub commit `9b8a09afdfe846b93e89f727ffc120ab567de427` (`feat(taskgraph): integrate DINO and GeoRSCLIP candidate filtering`, 2026-10-13 19:02 +08:00) was fast-forwarded into the local branch. It adds DINO/GeoRSCLIP candidate verification, shared tiled counting integration, prompt/config documentation, and 703 lines of focused regression tests. Planner spatial-cleaning changes and their partial dataset remain outside the competition-runtime commit.

## Production changes

- The Protocol 2.0 adapter routes two-image temporal requests first to the 4B ChangeChat provider, sends one-image inputs at or below 409,600 original pixels to direct 2B VLM, and sends larger single images to TaskGraph. The pixel threshold is configurable and separate from processor resizing, but the current value is only a conservative initial guard; it has not been calibrated against an image-quality/latency curve.
- The final answer contract now travels with the TaskGraph request. Integer answers require a `ScalarInt` or an exact integer response from the original images; BBOX answers accept one strict JSON box, validate the declared coordinate space, image index, and original-image bounds, then serialize only the requested contract fields.
- Runtime memory can only change an eligible high-resolution, single-image TaskGraph route to direct VLM when the matching record is explicitly verified. The old memory table has 3,738 rows; none carries enough role, correctness, and evaluation evidence for this whitelist, so 0 rows are retained. The verified table is empty, recording is disabled, and strict verification is enabled.
- TaskGraph gets at most one retry, and only for retryable transient failures while enough request budget remains. A TaskGraph execution or final-answer contract failure then gets one semantic-2B fallback using the original images. The configured request timeout is 120 s, with 20 s reserved for fallback, 6 s minimum to begin fallback, and 1 s response reserve. If fallback itself cannot produce a valid contract response, the service fails closed; it does not emit an invented answer.
- A shared Qwen3-VL manager now owns one 4B base and processor and serializes planner/ChangeChat adapter switches. It checks local adapter files and PEFT configuration compatibility, flushes registered KV sessions before switching, and exposes memory/switch metrics. Unit tests validate compatibility and cache invalidation. A real model load and GPU comparison have not been run, so memory savings and production readiness are unverified.
- Health responses expose lazy-load/readiness state. Startup preflight verifies configured local model paths for Planner, semantic 2B, and route 4B.

## Planner provenance and memory

The exact 2026-10-10 stable Planner adapter could not be identified. The local environment has no `QWEN3VL_4B_PLANNER_LORA` value, and the production example still expects deployment injection. The known local Sep 2 candidate has adapter SHA256 `defad37c62a53cb928d8879d8f690cc55806dffee7a81a547f121c9975511570`; the Oct 11 candidate has SHA256 `d7169cce768dcd01e86f1c2fd2847f7fddf85e45f9d61db3e23180a97211a4e6`. Neither establishes the requested Oct 10 lineage, so no model pointer was changed.

The memory audit source and evidence are under `scripts/taskgraph/audit_verified_memory.py` and `reports/taskgraph/runtime_memory_audit_20261013/`. The archived source table hash is `f649cf1b972dc06e339d8c17c69733f7ef43017ce8f6192399b2f0e23dee2cbe`. The new production allowlist contains zero rows.

## Routing, protocol, and CPU/GPU findings

The current routing order is: two-image temporal request → 4B direct route; one image at or below 409,600 original pixels → direct semantic 2B; larger single image → TaskGraph. Verified correctness memory can only relax the last route to direct 2B for a single image. Dataset names do not select the route. High-resolution questions are not routed by task wording.

The 2026-10-13 cloud Planner evaluator was confirmed to leave Transformers `BatchFeature`/`BatchEncoding` on CPU because the archived helper only recurses into plain dictionaries. In Transformers 5.14.1, generation then transfers full vocabulary logits back to CPU on every token. The production `Qwen3VLPlannerProvider` uses the native `.to(device)` path and does not share that defect. Production still has per-token CPU grammar checks; exact production overhead and an A/B speedup were not measured because the queued comparison was stopped before model loading.

Protocol contract, route, retry/fallback, BBOX, memory, planner-strategy, and shared-manager tests were run with the required Windows Python interpreter: **177 passed, 1 skipped, 1 deprecation warning**. The full repository suite has not yet completed. Ruff is unavailable in the current shell, so Ruff checks were skipped. No live competition endpoint test or real-model inference was run in this pass.

## Resource measurements

No per-task end-to-end latency, peak VRAM, or before/after shared-model VRAM measurement is available for these changes. Local hardware inspection showed an 8.6 GB class laptop GPU, which is insufficient evidence for safely loading the 4B model alongside the existing stack. The cloud coordination report records one historical evaluator snapshot (~13.5 CPU cores, ~34% instantaneous GPU utilization, 9.9 GB GPU memory), but it does not establish an optimized production latency baseline. No new timing or memory figures are inferred from these observations.

## Outstanding risks

- **P0:** The exact Oct 10 Planner adapter path/hash is unresolved; the deployment Planner pointer must remain deployment-provided until provenance is recovered.
- **P0:** Shared 4B adapter loading is structurally implemented but has no real PEFT/GPU smoke test. The runtime example enables sharing, so adapter/base compatibility and switch behavior must be smoke-tested on the target GPU before competition deployment.
- **P1:** 409,600 pixels is a provisional route threshold, not a measured quality boundary.
- **P1:** The verified memory allowlist is empty, so no legacy memory route is active.
- **P1:** Direct integer and BBOX VLM output contracts are strict; accuracy under the competition distribution has not been measured.
- **P1:** Retry/fallback can only run within the configured deadline; insufficient remaining time or invalid fallback output yields a protocol error.
- **P1:** The real challenge endpoint and full protocol check script were not available in this local pass; HTTP behavior has unit coverage only.

## CLIP teammate interface

The merged commit reuses `RegionRetrieverProvider.retrieve(RegionRetrievalRequest) -> RegionCandidates` in `src/sat_rs_vlm/taskgraph/providers.py`; its lower-level adapter seam remains `RetrieverProvider.score_regions(image_path, query, regions_xyxy) -> RetrievalResult` in `src/sat_rs_vlm/integrations/retrievers/protocol.py`. Preserve ordered one-score-per-region results, finite scores, latency/provider/model metadata, and the adapter/config registration. The commit's summary reports 195 fake-provider/regression checks and says real model/image tests were skipped; my combined post-merge regression run passed 209 tests with one skip.

## Planner spatial-cleaning handoff

The teammate's Planner spatial-semantic cleaning is a separate, incomplete workstream. Its current report describes 5,613 source rows, 1,857 released strict rows, 70 quarantined, and 3,686 pending; it also records unresolved defects in an earlier review batch and zero source images inspected. It must not be placed into the competition-runtime/GitHub CLIP commit or used as a replacement training set. Keep its working files separate for semantic re-audit.
