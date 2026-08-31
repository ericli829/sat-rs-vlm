# Open-source RS-CLIP provider survey

Checked on 2026-08-29 for the UHR Region Retriever. The goal is not image
classification accuracy; it is text-to-region ranking on the same candidate
tiles, with offline loading, batching, cacheability, and auditable provenance.

## Shortlist

| Priority | Project | Public code / weights | Actual inference surface | Fit for this module |
|---|---|---|---|---|
| P0 | Git-RSCLIP | [GitHub](https://github.com/Chen-Yang-Liu/Git-RSCLIP), [HF large](https://huggingface.co/lcybuaa/Git-RSCLIP), [HF base](https://huggingface.co/lcybuaa/Git-RSCLIP-base) | `transformers.AutoProcessor` + `AutoModel.get_image_features/get_text_features` | Best first real checkpoint. It matches the current lazy HuggingFace provider with minimal glue. |
| P0 | RemoteCLIP | [GitHub](https://github.com/ChenDelong1999/RemoteCLIP), [HF weights](https://huggingface.co/chendelong/RemoteCLIP) | OpenCLIP model plus `RemoteCLIP-{RN50,ViT-B-32,ViT-L-14}.pt`; encode and L2-normalize image/text features | Strong RS retrieval baseline and Apache-2.0 code. Needs an OpenCLIP backend because the official weights are not Transformers directories. |
| P0 | FarSLIP | [GitHub](https://github.com/NJU-LHRS/FarSLIP), [HF weights](https://huggingface.co/ZhenShiL/FarSLIP) | Custom `open_clip` fork, `.pt` checkpoints; supports image-text retrieval and fine-grained alignment | Good candidate for small/local objects and fine-grained captions. Needs isolated OpenCLIP environment/backend. MIT repository. |
| P1 | SkySense-O / SkySense-CLIP | [GitHub](https://github.com/zqcrafts/SkySense-O), [HF](https://huggingface.co/zqcraft/SkySense-O) | Custom SkySense-O code and checkpoint; broader open-vocabulary interpretation/segmentation system | Potentially strongest semantic/localization quality, but not a drop-in CLIP scorer. Treat as a separate provider experiment after P0. Apache-2.0 repository. |
| P1 | MPS-CLIP | [GitHub](https://github.com/Lcrucial1f/MPS-CLIP) | Custom OpenCLIP implementation and GeoRSCLIP/RS5M-style retrieval checkpoint | Relevant retrieval objective, but custom multi-perspective preprocessing makes it harder to guarantee identical tile semantics. MIT repository. |
| P1 | PriorCLIP | [GitHub](https://github.com/jaychempan/PriorCLIP) | OpenCLIP-derived `PIR` model; checkpoint linked through Baidu Disk | Retrieval-oriented and MIT code, but weight acquisition and custom architecture reduce reproducibility. |
| P2 | DGTRS Region-Phrase Alignment | [GitHub](https://github.com/Ali215666/DGTRS-Region-Phrase-Alignment) | LongCLIP-style local region/phrase alignment; research checkpoints are user-provided | Most conceptually aligned with region retrieval, but currently a training/evaluation project rather than a stable pretrained provider. MIT repository. |
| P2 | ST-GeoRSCLIP | [GitHub](https://github.com/YeYang12/ST-GeoRSCLIP-) | Spatiotemporal/geographic retrieval research code | Worth monitoring, but the public repository currently does not expose a simple, documented checkpoint path for our scorer. |

## Do not use as the first scorer

- [SatCLIP](https://github.com/microsoft/satclip) learns image-to-coordinate
  representations from Sentinel-2. Its core output is geographic location
  embedding, not a text-aligned image embedding, so it cannot implement
  `query -> candidate tile` without an additional text model.
- [RS-TransCLIP](https://github.com/elkhouryk/RS-TransCLIP) is a transductive
  improvement/evaluation method over existing CLIP/VLMs. It is useful for
  classification experiments but is not itself a standalone retrieval
  checkpoint/provider.
- Generic OpenAI CLIP/SigLIP remains a useful control, but should be labelled
  `generic_control`, not presented as an RS-specialized result.

## Compatibility with this repository

The current provider contract is `score_regions(image_path, query,
regions_xyxy) -> RetrievalResult`. The implementation in
`integrations/retrievers/clip.py` directly covers the Transformers-shaped P0
Git-RSCLIP path. RemoteCLIP and FarSLIP use OpenCLIP `.pt` weights, so they
must be loaded by a separate lazy OpenCLIP backend (or a sidecar) rather than
pretending that a `.pt` file is a Transformers model directory. Every backend
must preserve:

1. one score per input region in input order;
2. absolute original-image coordinates at the adapter boundary;
3. query/image batch encoding and score cache keys containing model identity;
4. `provider`, `model_id`, runtime, and peak-memory provenance.

## Recommended experiment order

1. Run Git-RSCLIP and RemoteCLIP on the same fixed tile manifest.
2. Add FarSLIP if its OpenCLIP environment is reproducible on the target GPU.
3. Compare generic SigLIP as a non-RS control.
4. Only then invest in SkySense-O, MPS-CLIP, PriorCLIP, or DGTRS adapters.

Use `scripts/retriever_benchmark.py` with identical `grid_size`, `top_k`,
coverage threshold, and Count gate threshold. Do not compare raw cosine values
across providers as if they were calibrated probabilities; compare ranking,
coverage, gate recall, latency, memory, and cache behavior.

## Current evidence boundary

This is a source/interface survey, not a quality leaderboard. No real
checkpoint is present in the current workspace, and no GPU is available in the
current environment. Therefore no Recall@K or VRAM number is claimed here;
those must be produced after downloading the checkpoints and running the fixed
manifest benchmark.
