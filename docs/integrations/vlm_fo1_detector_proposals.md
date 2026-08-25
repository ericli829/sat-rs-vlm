# VLM-FO1 detector proposal comparison

## Architecture

Detector proposal generation is deliberately separate from FO1 inference:

```text
evaluation JSONL (user question/image only)
        |
        v
extract_count_target_phrase(question)
        |
        +--> Grounding DINO provider (main runtime, local-only)
        +--> LAE-DINO sidecar (isolated MMDetection runtime)
        +--> cached ProposalResult
        |
        v
precomputed JSONL: bbox_list + bbox_scores + proposal_metadata
        |
        v
evaluate_vlm_fo1.py --proposal-backend precomputed
        |
        v
existing VLM-FO1 worker -> FO1 region parsing -> formal counting protocol
```

The provider interface always returns absolute pixel `xyxy` boxes, scores in
descending order, and deterministic top-k output.  Invalid or degenerate boxes
are filtered and counted in metadata.  Provider code never receives a reference
answer, assistant message, ground-truth count, or label.

On AutoDL, the expected detector root is `/root/autodl-fs/rs_detectors`; check
`/root/autodl-fs/rs_detectors/MANIFEST.txt` before configuring model and
checkpoint paths.  The repository does not copy these weights or add them to
Git.

`upn` remains the existing in-worker path.  `precomputed` is the common FO1
interface for Grounding DINO and LAE-DINO, so prompt, generation, parsing, and
formal metrics stay unchanged.

## Grounding DINO

The provider uses `AutoProcessor` and
`AutoModelForZeroShotObjectDetection` with `local_files_only=True`.  It loads
the local model once and reuses it.  Set `GROUNDING_DINO_MODEL` to the downloaded
directory; no Hugging Face fallback is allowed.

```bash
python scripts/integrations/precompute_vlm_fo1_proposals.py \
  --config configs/eval/vlm_fo1_grounding_dino.yaml \
  --input data/evaluation/tiers_v2/e_count_v2.jsonl \
  --output data/evaluation/proposals/grounding_dino_e_count_v2.jsonl \
  --provider grounding_dino \
  --model-path "$GROUNDING_DINO_MODEL" \
  --image-root "$VLM_FO1_IMAGE_ROOT" \
  --cache-dir "$VLM_FO1_PROPOSAL_CACHE_DIR"
```

The config records box/text thresholds, optional NMS, top-k, model identity, and
cache identity.  Do not tune thresholds on the formal evaluation set; use a
separate development subset and keep its selection manifest.

## LAE-DINO sidecar

LAE-DINO is an MMDetection/legacy stack and is never imported by the main
`rs-vlm` process.  Pass the exact config discovered from the downloaded source;
the integration intentionally does not guess a filename.

```bash
python environments/lae_dino/check_environment.py \
  --source-root /root/autodl-fs/rs_detectors/lae_dino/source/LAE-DINO \
  --discover
```

After selecting the exact config, validate the isolated environment and
precompute one of the three checkpoints:

```bash
python environments/lae_dino/check_environment.py \
  --source-root /root/autodl-fs/rs_detectors/lae_dino/source/LAE-DINO \
  --config "$LAE_DINO_CONFIG_LAE1M" \
  --checkpoint /root/autodl-fs/rs_detectors/lae_dino/checkpoints/lae_dino_swint_lae1m-28ca3a15.pth \
  --bert-root /root/autodl-fs/rs_detectors/lae_dino/weights/bert-base-uncased

python scripts/integrations/precompute_vlm_fo1_proposals.py \
  --config configs/eval/vlm_fo1_lae_dino_lae1m.yaml \
  --image-root "$VLM_FO1_IMAGE_ROOT" \
  --worker-python "$LAE_DINO_PYTHON" \
  --cache-dir "$VLM_FO1_PROPOSAL_CACHE_DIR"
```

Use `vlm_fo1_lae_dino_dior.yaml` and `vlm_fo1_lae_dino_dota.yaml` for the DIOR
and DOTA fine-tuned checkpoints.  LAE-DINO is a closed-set detector; its
target phrase is recorded for provenance but is not silently treated as open
vocabulary category filtering.  This comparability limitation must remain in
reports.

## Reusing proposal cache and running FO1

The cache key includes provider, model/checkpoint identity, image path/stat,
target phrase, thresholds, top-k, NMS, and schema version.  Re-running FO1 with
the same proposal JSONL therefore does not run the detector again.

```bash
python scripts/evaluation/evaluate_vlm_fo1.py \
  --config configs/eval/vlm_fo1_grounding_dino.yaml \
  --input data/evaluation/proposals/grounding_dino_e_count_v2.jsonl \
  --proposal-backend precomputed \
  --runtime-mode shared_rs_vlm \
  --attention-backend sdpa \
  --max-samples 5 \
  --output-dir reports/evaluation/vlm_fo1/grounding_dino_smoke5
```

The evaluator records proposal provider/model/metadata, proposal latency, FO1
latency, total latency, proposal failures, zero-proposal rate, proposal count
distribution, parse rate, exact accuracy, within-1 accuracy, and MAE.

For a bounded detector smoke, add `--max-samples 5` (or `10`) to the
precompute command and pass `--expected-population 5` to the evaluator.  The
formal configs intentionally keep the unified-v2 population guard at 377.

## Visualization

```bash
python scripts/integrations/visualize_vlm_fo1_proposals.py \
  --image /path/to/image.png \
  --proposal-json data/evaluation/proposals/grounding_dino_e_count_v2.jsonl \
  --sample-id SAMPLE_ID \
  --output reports/proposals/SAMPLE_ID.png
```

## Comparison order

Run mock/unit tests first, then 5--10 sample Grounding DINO and one LAE-DINO
checkpoint smoke.  Inspect visualizations and compare the three LAE checkpoints
on the same bounded sample before any full evaluation.  Full formal evaluation
must use fixed detector settings and separate proposal files; do not search
thresholds against the formal references.
