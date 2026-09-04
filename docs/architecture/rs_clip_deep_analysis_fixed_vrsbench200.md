# Deep RS-CLIP evaluation on corrected VRSBench-200

All models use the same corrected 200 annotations, 3x3 grid, Top-5, category query, coverage threshold 0.5, and CPU.

## Dataset audit

- 200 annotations from 127 unique images; 122 rows belong to repeated images (maximum 4 rows/image).
- 79 GT boxes changed after fixing normalized-coordinate scaling and clipping.
- Size distribution: small(0.5-2%): 44, large(>=10%): 70, medium(2-10%): 47, tiny(<0.5%): 39.
- Confidence intervals use image-cluster bootstrap; annotations from the same image are resampled together.

## Overall ranking

| Rank | Model | R@1 | R@3 | R@5 | MRR | AP | NDCG@5 | Mean coverage | Random R@5 | Oracle R@5 | Normalized gain |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | RemoteCLIP-ViT-B-32 | 34.5% | 58.0% | 64.0% | 0.471 | 0.663 | 0.713 | 63.7% | 39.4% | 71.0% | 0.778 |
| 2 | GeoRSCLIP-ViT-B-32 | 31.5% | 53.0% | 64.0% | 0.446 | 0.629 | 0.686 | 64.7% | 39.4% | 71.0% | 0.778 |
| 3 | FarSLIP1_ViT-B-32 | 30.5% | 53.5% | 63.0% | 0.437 | 0.616 | 0.672 | 63.3% | 39.4% | 71.0% | 0.746 |
| 4 | SatelliteCLIP | 30.5% | 53.0% | 59.0% | 0.435 | 0.613 | 0.649 | 60.1% | 39.4% | 71.0% | 0.620 |
| 5 | Git-RSCLIP-base | 12.0% | 35.5% | 45.0% | 0.277 | 0.390 | 0.414 | 47.6% | 39.4% | 71.0% | 0.176 |

## Recall@5 by target size

| Model | Tiny | Small | Medium | Large |
|---|---:|---:|---:|---:|
| RemoteCLIP-ViT-B-32 | 76.9% | 97.7% | 76.6% | 27.1% |
| GeoRSCLIP-ViT-B-32 | 84.6% | 90.9% | 76.6% | 27.1% |
| FarSLIP1_ViT-B-32 | 82.1% | 88.6% | 74.5% | 28.6% |
| SatelliteCLIP | 64.1% | 90.9% | 72.3% | 27.1% |
| Git-RSCLIP-base | 64.1% | 77.3% | 48.9% | 11.4% |
| Random Top-5 | 55.6% | 55.6% | 44.9% | 16.7% |
| Oracle grid limit | 100.0% | 100.0% | 80.9% | 30.0% |

## Pairwise inference

Recall deltas use image-cluster bootstrap. Image-cluster permutation p-values are adjusted across all ten model pairs with Holm's method; row-level exact McNemar results remain in the JSON audit.

| A vs B | R@5 delta | Cluster 95% CI | Coverage W/T/L | Cluster Holm p |
|---|---:|---:|---:|---:|
| Git-RSCLIP-base vs RemoteCLIP-ViT-B-32 | -19.0 pp | [-25.7, -13.0] pp | 19/109/72 | 0.0005 |
| Git-RSCLIP-base vs FarSLIP1_ViT-B-32 | -18.0 pp | [-25.4, -11.1] pp | 17/107/76 | 0.0005 |
| Git-RSCLIP-base vs GeoRSCLIP-ViT-B-32 | -19.0 pp | [-26.3, -11.9] pp | 15/108/77 | 0.0005 |
| Git-RSCLIP-base vs SatelliteCLIP | -14.0 pp | [-21.8, -6.9] pp | 24/104/72 | 0.0031 |
| RemoteCLIP-ViT-B-32 vs FarSLIP1_ViT-B-32 | +1.0 pp | [-4.6, +6.7] pp | 21/159/20 | 1.0000 |
| RemoteCLIP-ViT-B-32 vs GeoRSCLIP-ViT-B-32 | +0.0 pp | [-4.8, +4.9] pp | 18/160/22 | 1.0000 |
| RemoteCLIP-ViT-B-32 vs SatelliteCLIP | +5.0 pp | [+0.0, +10.4] pp | 28/154/18 | 0.4845 |
| FarSLIP1_ViT-B-32 vs GeoRSCLIP-ViT-B-32 | -1.0 pp | [-4.3, +2.0] pp | 5/185/10 | 1.0000 |
| FarSLIP1_ViT-B-32 vs SatelliteCLIP | +4.0 pp | [+0.0, +8.1] pp | 20/167/13 | 0.4845 |
| GeoRSCLIP-ViT-B-32 vs SatelliteCLIP | +5.0 pp | [+0.5, +9.6] pp | 22/167/11 | 0.3474 |

## Isolated CPU latency

Each model ran alone in a fresh process. Three rows were used for warm-up,
followed by 20 measured rows. Model loading is excluded from the measured
steady-state latency.

| Model | Mean (ms/image) | Median | P90 | P95 |
|---|---:|---:|---:|---:|
| GeoRSCLIP-ViT-B-32 | 660 | **653** | **711** | **715** |
| RemoteCLIP-ViT-B-32 | 665 | 648 | 718 | 722 |
| FarSLIP1_ViT-B-32 | 667 | 655 | 718 | 729 |
| SatelliteCLIP | 703 | 707 | 739 | 741 |
| Git-RSCLIP-base | 13,092 | 13,088 | 13,266 | 13,426 |

Raw latency reports are `reports/evaluation/latency_*_cpu20.json`.

## Final recommendation

- Use RemoteCLIP as the default 3x3 retriever: it ties for best R@5, leads R@1,
  MRR, AP, and NDCG@5, and has sub-second isolated CPU median latency.
- Keep GeoRSCLIP as the strongest alternate. It ties RemoteCLIP on R@5 and has
  slightly better mean GT coverage, but the pairwise difference is not
  statistically reliable.
- Do not select Git-RSCLIP for this pipeline under the tested protocol. It is
  below the analytical random Top-5 baseline on R@5 and roughly 20x slower.
- Improve candidate generation before further model tuning. The fixed 3x3
  single-cell Oracle is only 71% overall and 30% for large targets; overlapping
  or multi-cell/multiscale proposals are the next high-value experiment.

## Interpretation guardrails

- Compare model R@5 against the random and Oracle baselines, not against zero.
- Overlapping pairwise cluster intervals or Holm-adjusted p >= 0.05 do not support a statistically reliable superiority claim.
- Parallel CPU runs are suitable for quality comparison but not strict latency ranking; latency requires isolated warm-up runs.
