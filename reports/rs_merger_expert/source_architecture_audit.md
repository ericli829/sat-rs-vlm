# Qwen3-VL R1 source architecture audit

- Status: **blocked — exact local model assets are absent**.
- Runtime available locally: Python 3.11.9, torch 2.6.0+cu126, Transformers 5.13.0,
  PEFT 0.19.1, CUDA on an RTX 4060 Laptop GPU.
- No Qwen3-VL-4B directory, formal R1 adapter/manifest, or R1 visual sidecar is configured in this
  workspace or its environment.

Consequently, block count, hidden sizes, merger paths/shapes/counts, DeepStack indexes/injection
order, layer 0-3 q/k/v/o paths, and R1 merge parity are intentionally not guessed. Run
`scripts/training/audit_rs_merger_architecture.py` against the exact AutoDL model/R1 assets. The
script overwrites this report and exits non-zero on any 24/1024/2560/2/[5,11,17] mismatch.

Source-only inspection of the installed Transformers 5.13.0 implementation confirms that the main
merger uses pre-shuffle LayerNorm while DeepStack mergers use post-shuffle LayerNorm, and that each
DeepStack residual is applied after its decoder layer. The official 4B config yields derived (not
runtime-audited) trainable counts C1=109,105,152, C2=24,160,256, and C3=25,470,976, of which
1,310,720 are interface LoRA parameters.

Formal expert training is not authorized by this local report.
