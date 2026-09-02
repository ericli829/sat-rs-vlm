# Counting System vs TaskGraph contracts

`04_counting_system_plan` is the counting **algorithm backend / research workspace**.
It owns multi-scale tiling (Global / Native / Fine), core ownership, same-scale NMS,
and cross-scale fusion. Do not treat its internal `counting_system.runtime` types as
the TaskGraph public contract.

`src/sat_rs_vlm/taskgraph` is the **authoritative runtime contract**. COUNT exchanges
only `sat_rs_vlm.taskgraph.runtime_types` objects, and the operator output is always
`ScalarInt`.

```text
TaskGraph COUNT(EntitySet | valid SelectResult)
        → cardinality
        → ScalarInt

TaskGraph COUNT(ImageRef | Region)
        → CountingProvider.count(CountingRequest(scope, target, entire))
        → counting_system tiling / fusion
        → isolated LAE sidecar (lae_dino_lae1m, not tiled(...))
        → original-image XYXY
        → CountingResult
        → ScalarInt
```

LOCATE continues to use `RuntimeProviders.detection`. COUNT uses
`RuntimeProviders.counting`. Do not register counting_system as a global
`DetectionProvider`.
